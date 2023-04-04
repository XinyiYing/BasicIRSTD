###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
import torch.nn.functional as F

# import
# from .base import BaseNet
# from .fcn import FCNHead

# class reconet(BaseNet):
#     def __init__(self, nclass, backbone, aux=True, se_loss=True, norm_layer=nn.BatchNorm2d, dim=512, **kwargs):
#         super(reconet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
#         self.head = reconetHead(2048, nclass, norm_layer, dim, se_loss=se_loss, up_kwargs=self._up_kwargs)
#         if aux:
#             self.auxlayer = FCNHead(1024, nclass, norm_layer)
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         _, c2, c3, c4 = self.base_forward(x)
#
#         x = list(self.head(c4))
#         x[0] = upsample(x[0], (h,w), **self._up_kwargs)
#
#         if self.aux:
#             auxout = self.auxlayer(c3)
#             auxout = upsample(auxout, (h,w), **self._up_kwargs)
#             x.append(auxout)
#
#         return tuple(x)

class TGMandTRM(nn.Module):
    def __init__(self, h, norm_layer=None):
        super(TGMandTRM, self).__init__()
        self.rank = 16
        self.ps = [1, 1, 1, 1]
        self.h = h
        conv1_1, conv1_2, conv1_3 = self.ConvGeneration(self.rank, h)

        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3

        # self.lam = torch.ones(self.rank, requires_grad=True, device='cuda')
        self.lam = nn.Parameter(torch.ones(self.rank), requires_grad=True)

        self.pool = nn.AdaptiveAvgPool2d(self.ps[0])

        # self.fusion = nn.Sequential(
        #     nn.Conv2d(512, 512, 1, padding=0, bias=False),
        #     norm_layer(512),
        #     # nn.Sigmoid(),
        #     nn.ReLU(True),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')


    def forward(self, x):
        b, c, height, width = x.size()
        C = self.pool(x)
        H = self.pool(x.permute(0, 3, 1, 2).contiguous())
        W = self.pool(x.permute(0, 2, 3, 1).contiguous())
        # self.lam = F.softmax(self.lam,-1)
        lam = torch.chunk(F.softmax(self.lam, -1), dim=0, chunks=self.rank)
        list = []
        for i in range(0, self.rank):
            list.append(lam[i]*self.TukerReconstruction(b, self.h , self.ps[0], self.conv1_1[i](C), self.conv1_2[i](H), self.conv1_3[i](W)))
        tensor1 = sum(list)
        tensor1 = torch.cat((x , F.relu_(x * tensor1)), 1)
        return tensor1

    def ConvGeneration(self, rank, h):
        conv1 = []
        n = 1
        for _ in range(0, rank):
                conv1.append(nn.Sequential(
                nn.Conv2d(128, 128 // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, rank):
                conv2.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, rank):
                conv3.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv3 = nn.ModuleList(conv3)

        return conv1, conv2, conv3

    def TukerReconstruction(self, batch_size, h, ps, feat, feat2, feat3):
        b = batch_size
        C = feat.view(b, -1, ps)
        H = feat2.view(b, ps, -1)
        W = feat3.view(b, ps * ps, -1)
        CHW = torch.bmm(torch.bmm(C, H).view(b, -1, ps * ps), W).view(b, -1, h, h)
        return CHW

class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        norm_layer = nn.BatchNorm1d if isinstance(norm_layer, nn.BatchNorm2d) else \
            encoding.nn.BatchNorm1d
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            encoding.nn.Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)

class reconetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer,  dim,  se_loss=True, up_kwargs=None):
        super(reconetHead, self).__init__()
        inter_channels = in_channels // 4
        h = dim // 8
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1, dilation=1, padding=0, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True))

        self.decomp =TGMandTRM(h=h, norm_layer=norm_layer)
        # self.decomp = lambda x:x

        # self.encmodule = EncModule(512, out_channels, ncodes=32,
        #                            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(
                                   nn.Conv2d(128*2, inter_channels, 1, padding=0, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   # nn.Dropout2d(0.05, False),
                                   nn.Conv2d(inter_channels, out_channels, 1),
                                   )

    def forward(self, x):
        # feat_outs = []
        feat = self.feat(x)
        # feat_outs = list(self.encmodule(feat))
        outs = self.decomp(feat)
        # feat_outs[0] = self.conv6(torch.cat((outs, feat_outs[0]), 1))
        outs = self.conv6(outs)
        # feat_outs.append(outs)
        return outs

def get_reconet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'pcontext': 'pcontext',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = reconet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('reconet_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_reconet_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""reconet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_reconet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_reconet('ade20k', 'resnet50', pretrained, root=root, **kwargs)


if __name__ == '__main__':
    from thop import profile

    model = get_reconet_resnet50_ade()
    model.to('cuda')
    input = torch.randn(1, 1, 512, 640)
    input = input.to('cuda')
    flops, params = profile(model, inputs=(input, ))
    print(flops, params)