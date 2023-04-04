from __future__ import division
import os
from torch.nn.modules import module
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from .fusion import AsymBiChaFuse

# from mxnet import nd
from torchvision import transforms
from  torchvision.models.resnet import BasicBlock

class _FCNHead(nn.Module):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels, momentum=momentum),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels=inter_channels, out_channels=channels,kernel_size=1)
        )
    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.block(x)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ASKCResNetFPN(nn.Module):
    def __init__(self, in_channels=1, layers=[4,4,4], channels=[8,16,32,64], fuse_mode='AsymBi', act_dilation=16, classes=1, tinyFlag=False,
                 norm_layer=BatchNorm2d,groups=1,norm_kwargs=None, **kwargs):
        super(ASKCResNetFPN, self).__init__()

        self.layer_num = len(layers)
        self.tinyFlag = tinyFlag
        self.groups = groups
        self._norm_layer = norm_layer
        stem_width = int(channels[0])
        self.momentum=0.9
        if tinyFlag:
            self.stem = nn.Sequential(
                norm_layer(in_channels, self.momentum),
                nn.Conv2d(in_channels, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True)
            )

        else:
            self.stem = nn.Sequential(
                # self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=2,
                #                          padding=1, use_bias=False))
                # self.stem.add(norm_layer(in_channels=stem_width*2))
                # self.stem.add(nn.Activation('relu'))
                # self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
                norm_layer(in_channels, momentum=self.momentum),
                nn.Conv2d(in_channels=in_channels, out_channels=stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=stem_width, out_channels=stem_width, kernel_size=3, stride=1, padding=1,
                          bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=stem_width, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1,
                          bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )


            # self.head1 = _FCNHead(in_channels=channels[1], channels=classes)
            # self.head2 = _FCNHead(in_channels=channels[2], channels=classes)
            # self.head3 = _FCNHead(in_channels=channels[3], channels=classes)
            # self.head4 = _FCNHead(in_channels=channels[4], channels=classes)

            self.head = _FCNHead(in_channels=channels[0], channels=classes, momentum=self.momentum)

            self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                           out_channels=channels[1],
                                           in_channels=channels[1], stride=1)

            self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                           out_channels=channels[2], stride=2,
                                           in_channels=channels[1])
            #
            self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                           out_channels=channels[3], stride=2,
                                           in_channels=channels[2])
            self.deconv2 = nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=(4, 4),
                                              ##channels: 8 16 32 64
                                              stride=2, padding=1)
            self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                             out_channels=channels[2], stride=1,
                                             in_channels=channels[2])


            self.deconv1 = nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=(4, 4),
                                              stride=2, padding=1)

            self.deconv0 = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=(4, 4),
                                              stride=2, padding=1)

            self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                             out_channels=channels[1], stride=1,
                                             in_channels=channels[1])


            if self.layer_num == 4:
                self.layer4 = self._make_layer(block=BasicBlock, blocks=layers[3],
                                               out_channels=channels[3], stride=2,
                                               in_channels=channels[3])

            if self.layer_num == 4:
                self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[3])  # channels[4]

            self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[2])  # 64
            self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[1])  # 32

            # if fuse_order == 'reverse':
            #     self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[2])  # channels[2]
            #     self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[3])  # channels[3]
            #     self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]
            # elif fuse_order == 'normal':
            # self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]
            # self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]
            # self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]

    def _make_layer(self, block, out_channels, in_channels, blocks, stride):

        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels , stride),
                norm_layer(out_channels * block.expansion, momentum=self.momentum),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.groups, norm_layer=norm_layer))
        self.inplanes = out_channels  * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, self.groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _fuse_layer(self, fuse_mode, channels):

        if fuse_mode == 'AsymBi':
          fuse_layer = AsymBiChaFuse(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')
        return fuse_layer

    def forward(self, x):

        _, _, hei, wid = x.shape# 1024 1024

        x = self.stem(x)      #torch.Size([8, 16, 256, 256])
        c1 = self.layer1(x)   # torch.Size([8, 16, 256, 256])
        c2 = self.layer2(c1)  # torch.Size([8, 32, 128, 128])

        out = self.layer3(c2)  # (8,64, 64, 64)

        if self.layer_num == 4:
            c4 = self.layer4(out) # torch.Size([8,64, 32, 32])
            if self.tinyFlag:
                c4 = transforms.Resize([hei//4, wid//4])(c4)  # down 4
            else:
                c4 =  transforms.Resize([hei//16, wid//16])(c4)  # down 16 torch.Size([8, 64, 64, 64])
            out = self.fuse34(c4, out) #torch.Size([8, 64, 128, 128])`

        if self.tinyFlag:
            out =  transforms.Resize([hei//2, wid//2])(out)  # down 16 torch.Size([8, 64, 64, 64])
        else:
            out =  transforms.Resize([hei//16, wid//16])(out)    # down 8, 128 torch.Size([8, 64, 64, 64])

        out = self.deconv2(out) # torch.Size([8, 32, 128, 128])
        out = self.fuse23(out, c2) # torch.Size([8, 32, 128, 128])
        if self.tinyFlag:
            out =  transforms.Resize([hei, wid])(out)  # down 1
        else:
            out =  transforms.Resize( [hei//8, wid//8])(out)  # (4,16,120,120)

        out = self.deconv1(out)  # torch.Size([8, 16, 256, 256])
        out = self.fuse12(out, c1) # torch.Size([8, 16, 256, 256])

        out = self.deconv0(out)  # torch.Size([8, 8, 512, 512])
        pred = self.head(out) # torch.Size([8, 8, 512, 512])


        if self.tinyFlag:
            out = pred
        else:
            out = transforms.Resize( [hei, wid])(pred)  # down 4

        ######### reverse order ##########
        # up_c2 = F.contrib.BilinearResize2D(c2, height=hei//4, width=wid//4)  # down 4
        # fuse2 = self.fuse12(up_c2, c1)  # down 4, channels[2]
        #
        # up_c3 = F.contrib.BilinearResize2D(c3, height=hei//4, width=wid//4)  # down 4
        # fuse3 = self.fuse23(up_c3, fuse2)  # down 4, channels[3]
        #
        # up_c4 = F.contrib.BilinearResize2D(c4, height=hei//4, width=wid//4)  # down 4
        # fuse4 = self.fuse34(up_c4, fuse3)  # down 4, channels[4]
        #

        ######### normal order ##########
        # out = F.contrib.BilinearResize2D(c4, height=hei//16, width=wid//16)
        # out = self.fuse34(out, c3)
        # out = F.contrib.BilinearResize2D(out, height=hei//8, width=wid//8)
        # out = self.fuse23(out, c2)
        # out = F.contrib.BilinearResize2D(out, height=hei//4, width=wid//4)
        # out = self.fuse12(out, c1)
        # out = self.head(out)
        # out = F.contrib.BilinearResize2D(out, height=hei, width=wid)


        return out.sigmoid()

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)




# class BasicContextNet(HybridBlock):
#     def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1,
#                  conv_mode='xxx', act_type='relu', skernel=3, act_dilation=16,
#                  useReLU=False, use_act_head=False, check_fullly=False, act_layers=4,
#                  act_order='xxx', asBackbone=False, addstem=False, maxpool=True, **kwargs):
#         super(BasicContextNet, self).__init__(**kwargs)
#         assert act_type in ['swish', 'prelu', 'relu', 'xUnit', 'SeqATAC', 'SpaATAC', 'ChaATAC',
#                             'MSSeqATAC', 'MSSeqATACAdd', 'MSSeqATACConcat'], "Unknown act_type"
#         assert conv_mode in ['fixed', 'learned', 'ChaDyReF', 'SeqDyReF', 'SK_ChaDyReF',
#                              'SK_1x1DepthDyReF', 'SK_MSSpaDyReF', 'SK_SpaDyReF',
#                              'Direct_Add', 'SKCell', 'SK_SeqDyReF', 'Sub_MSSpaDyReF',
#                              'SK_MSSeqDyReF', 'iAAMSSpaDyReF'], \
#             "Unknown conv_mode"
#         # stem_width = int(channels // 2)
#         with self.name_scope():
#             self.features = nn.HybridSequential(prefix='')
#             if addstem:
#                 self.features.add(nn.Conv2D(channels=channels, kernel_size=3, strides=2,
#                                             padding=1, use_bias=False))
#                 self.features.add(nn.BatchNorm(in_channels=channels))
#                 self.features.add(nn.Activation('relu'))
#                 self.features.add(nn.Conv2D(channels=channels, kernel_size=3, strides=1,
#                                             padding=1, use_bias=False))
#                 self.features.add(nn.BatchNorm(in_channels=channels))
#                 self.features.add(nn.Activation('relu'))
#                 self.features.add(nn.Conv2D(channels=channels*2, kernel_size=3, strides=1,
#                                          padding=1, use_bias=False))
#                 self.features.add(nn.BatchNorm(in_channels=channels*2))
#                 self.features.add(nn.Activation('relu'))
#             if maxpool:
#                 self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
#
#             for i, dilation in enumerate(dilations):
#                 self.features.add(self._make_layer(
#                     dilation=dilation, channels=channels, stage_index=i, conv_mode=conv_mode,
#                     act_type=act_type, skernel=skernel, act_dilation=act_dilation,
#                     useReLU=useReLU, check_fullly=check_fullly, act_layers=act_layers,
#                     act_order=act_order, asBackbone=asBackbone))
#             if use_act_head:
#                 self.head = ATAC_FCNHead(head_act=act_type, useReLU=useReLU,
#                                          in_channels=channels, channels=classes)
#             else:
#                 self.head = _FCNHead(in_channels=channels, channels=classes)
#
#     def _make_layer(self, dilation, channels, stage_index, conv_mode, act_type, skernel,
#                     act_dilation, useReLU, check_fullly, act_layers, act_order, asBackbone):
#         layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
#         with layer.name_scope():
#
#             if check_fullly:
#                 if act_order == 'bac':
#                     # 后面的层优先用 Attention
#                     if stage_index + act_layers < 5:
#                         act_type = 'relu'
#                 elif act_order == 'pre':
#                  # 前面的层优先用 Attention
#                     if act_layers - stage_index - 1 < 0:
#                         act_type = 'relu'
#                 else:
#                     raise ValueError('Unknown act_order')
#
#             if conv_mode == 'fixed':
#
#                 layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
#                                     padding=dilation))
#                 layer.add(nn.BatchNorm())
#
#                 if act_type == 'prelu':
#                     layer.add(nn.PReLU())
#                 elif act_type == 'relu':
#                     layer.add(nn.Activation('relu'))
#                 elif act_type == 'swish':
#                     layer.add(nn.Swish())
#                 elif act_type == 'xUnit':
#                     layer.add(xUnit(channels=channels, skernel_size=5))
#                 elif act_type == 'SpaATAC':
#                     layer.add(SpaATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                       useReLU=useReLU, asBackbone=asBackbone))
#                 elif act_type == 'ChaATAC':
#                     layer.add(ChaATAC(channels=channels, useReLU=useReLU, useGlobal=False,
#                                       asBackbone=asBackbone))
#                 elif act_type == 'SeqATAC':
#                     layer.add(SeqATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                       useReLU=useReLU, asBackbone=asBackbone))
#                     # layer.add(DilatedSeqATACBackbone(channels=channels, dilation=act_dilation))
#                 elif act_type == 'MSSeqATAC':
#                     layer.add(MSSeqATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                         useReLU=useReLU, asBackbone=asBackbone))
#                     # layer.add(DilatedSeqATACBackbone(channels=channels, dilation=act_dilation))
#                 elif act_type == 'MSSeqATACAdd':
#                     layer.add(MSSeqATACAdd(skernel=skernel, channels=channels,
#                                            dilation=act_dilation, useReLU=useReLU,
#                                            asBackbone=asBackbone))
#                 elif act_type == 'MSSeqATACConcat':
#                     layer.add(MSSeqATACConcat(skernel=skernel, channels=channels,
#                                            dilation=act_dilation, useReLU=useReLU,
#                                            asBackbone=asBackbone))
#                 else:
#                     raise ValueError('Unknown act_type')
#
#             elif conv_mode == 'learned':
#                 layer.add(LearnedCell(channels=channels, dilations=dilation))
#             elif conv_mode == 'ChaDyReF':
#                 layer.add(ChaDyReFCell(channels=channels, dilations=dilation))
#             elif conv_mode == 'SK_ChaDyReF':
#                 layer.add(SK_ChaDyReFCell(channels=channels, dilations=dilation))
#             elif conv_mode == 'SK_1x1DepthDyReF':
#                 layer.add(SK_1x1DepthDyReFCell(channels=channels, dilations=dilation))
#             elif conv_mode == 'SK_MSSpaDyReF':
#                 layer.add(SK_MSSpaDyReFCell(channels=channels, dilations=dilation,
#                                             act_dilation=act_dilation,
#                                             asBackbone=asBackbone))
#             elif conv_mode == 'iAAMSSpaDyReF':
#                 layer.add(iAAMSSpaDyReFCell(channels=channels, dilations=dilation,
#                                             asBackbone=asBackbone))
#             elif conv_mode == 'SK_MSSeqDyReF':
#                 layer.add(SK_MSSeqDyReFCell(channels=channels, dilations=dilation,
#                                             asBackbone=asBackbone))
#             elif conv_mode == 'Sub_MSSpaDyReF':
#                 layer.add(Sub_MSSpaDyReFCell(channels=channels, dilations=dilation,
#                                             asBackbone=asBackbone))
#
#             elif conv_mode == 'Direct_Add':
#                 layer.add(Direct_AddCell(channels=channels, dilations=dilation,
#                                             asBackbone=asBackbone))
#             elif conv_mode == 'SK_SpaDyReF':
#                 layer.add(SK_SpaDyReFCell(channels=channels, dilations=dilation,
#                                           act_dilation=act_dilation))
#             elif conv_mode == 'SKCell':
#                 layer.add(SKCell(channels=channels, dilations=dilation))
#             elif conv_mode == 'SeqDyReF':
#                 layer.add(SeqDyReFCell(channels=channels, dilations=dilation,
#                                        act_dilation=act_dilation, useReLU=useReLU,
#                                        asBackbone=asBackbone))
#             elif conv_mode == 'SK_SeqDyReF':
#                 layer.add(SK_SeqDyReFCell(channels=channels, dilations=dilation,
#                                        act_dilation=act_dilation, useReLU=useReLU,
#                                        asBackbone=asBackbone))
#             elif conv_mode == 'dynamic':
#                 layer.add(DynamicCell(channels=channels, dilations=dilation))
#             else:
#                 raise ValueError('Unknown conv_mode')
#         return layer
#
#     def hybrid_forward(self, F, x):
#
#         _, _, hei, wid = x.shape
#         x = self.features(x)
#         x = self.head(x)
#
#         out = F.contrib.BilinearResize2D(x, height=hei, width=wid)
#
#         return out
#
#     def evaluate(self, x):
#         """evaluating network with inputs and targets"""
#         return self.forward(x)


# class ATAC_FCNHead(BasicBlock):
#     # pylint: disable=redefined-outer-name
#     def __init__(self, head_act, useReLU, in_channels, channels, norm_layer=nn.BatchNorm,
#                  norm_kwargs=None, **kwargs):
#         super(ATAC_FCNHead, self).__init__()
#         with self.name_scope():
#             self.block = nn.HybridSequential()
#             inter_channels = in_channels // 4
#             with self.block.name_scope():
#                 self.block.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels,
#                                          kernel_size=3, padding=1, use_bias=False))
#                 self.block.add(norm_layer(in_channels=inter_channels,
#                                           **({} if norm_kwargs is None else norm_kwargs)))
#                 # self.block.add(nn.Activation('relu'))
#
#                 if head_act == 'prelu':
#                     self.block.add(nn.PReLU())
#                 elif head_act == 'relu':
#                     self.block.add(nn.Activation('relu'))
#                 elif head_act == 'xUnit':
#                     self.block.add(xUnit(channels=inter_channels))
#                 elif head_act == 'SpaATAC':
#                     self.block.add(SpaATAC(skernel=3, channels=inter_channels, dilation=1,
#                                            useReLU=useReLU))
#                 elif head_act == 'ChaATAC':
#                     self.block.add(ChaATAC(channels=inter_channels, useReLU=useReLU,
#                                            useGlobal=False))
#                 elif head_act == 'SeqATAC':
#                     self.block.add(SeqATAC(skernel=3, channels=inter_channels, dilation=1,
#                                            useReLU=useReLU))
#                     # layer.add(DilatedSeqATACBackbone(channels=channels, dilation=act_dilation))
#                 else:
#                     raise ValueError('Unknown act_type')
#
#                 self.block.add(nn.Dropout(0.1))
#                 self.block.add(nn.Conv2D(in_channels=inter_channels, channels=channels,
#                                          kernel_size=1))
#
#     # pylint: disable=arguments-differ
#     def hybrid_forward(self, F, x):
#         return self.block(x)
#
#
# class DyRefNet(HybridBlock):
#     def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1,
#                  act_type='relu', skernel=3, act_dilation=16, useReLU=False,
#                  use_act_head=False, check_fullly=False, act_layers=4, act_order='xxx',
#                  asBackbone=False, **kwargs):
#         super(DyRefNet, self).__init__(**kwargs)
#         assert act_type in ['prelu', 'relu', 'xUnit', 'SeqATAC', 'SpaATAC', 'ChaATAC', 'MSSeqATAC'], \
#             "Unknown act_type"
#         with self.name_scope():
#             self.features = nn.HybridSequential(prefix='')
#             for i, dilation in enumerate(dilations):
#                 self.features.add(self._make_layer(
#                     dilation=dilation, channels=channels, stage_index=i, act_type=act_type,
#                     skernel=skernel, act_dilation=act_dilation, useReLU=useReLU,
#                     check_fullly=check_fullly, act_layers=act_layers, act_order=act_order,
#                     asBackbone=asBackbone))
#             if use_act_head:
#                 self.head = ATAC_FCNHead(head_act=act_type, useReLU=useReLU,
#                                          in_channels=channels, channels=classes)
#             else:
#                 self.head = _FCNHead(in_channels=channels, channels=classes)
#
#     def _make_layer(self, dilation, channels, stage_index, act_type, skernel,
#                     act_dilation, useReLU, check_fullly, act_layers, act_order, asBackbone):
#         layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
#         with layer.name_scope():
#             layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
#                                 padding=dilation))
#             layer.add(nn.BatchNorm())
#
#             if check_fullly:
#                 if act_order == 'bac':
#                     # 后面的层优先用 Attention
#                     if stage_index + act_layers < 4:
#                         act_type = 'relu'
#                 elif act_order == 'pre':
#                  # 前面的层优先用 Attention
#                     if act_layers - stage_index - 1 < 0:
#                         act_type = 'relu'
#                 else:
#                     raise ValueError('Unknown act_order')
#
#             if act_type == 'prelu':
#                 layer.add(nn.PReLU())
#             elif act_type == 'relu':
#                 layer.add(nn.Activation('relu'))
#             elif act_type == 'xUnit':
#                 layer.add(xUnit(channels=channels, skernel_size=5))
#             elif act_type == 'SpaATAC':
#                 layer.add(SpaATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                   useReLU=useReLU, asBackbone=asBackbone))
#             elif act_type == 'ChaATAC':
#                 layer.add(ChaATAC(channels=channels, useReLU=useReLU, useGlobal=False,
#                                   asBackbone=asBackbone))
#             elif act_type == 'SeqATAC':
#                 layer.add(SeqATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                   useReLU=useReLU, asBackbone=asBackbone))
#                 # layer.add(DilatedSeqATACBackbone(channels=channels, dilation=act_dilation))
#             elif act_type == 'MSSeqATAC':
#                 layer.add(MSSeqATAC(skernel=skernel, channels=channels,
#                                     dilation=act_dilation, useReLU=useReLU,
#                                     asBackbone=asBackbone))
#                 # layer.add(DilatedSeqATACBackbone(channels=channels, dilation=act_dilation))
#             else:
#                 raise ValueError('Unknown act_type')
#         return layer
#
#     def hybrid_forward(self, F, x):
#         x = self.features(x)
#         x = self.head(x)
#
#         return x


# class VisualBasicContextNet(HybridBlock):
#     def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1,
#                  conv_mode='xxx', act_type='relu', skernel=3, act_dilation=16,
#                  useReLU=False, use_act_head=False, check_fullly=False, act_layers=4,
#                  act_order='xxx', asBackbone=False, addstem=False, **kwargs):
#         super(VisualBasicContextNet, self).__init__(**kwargs)
#         assert act_type in ['swish', 'prelu', 'relu', 'xUnit', 'SeqATAC', 'SpaATAC', 'ChaATAC',
#                             'MSSeqATAC', 'MSSeqATACAdd', 'MSSeqATACConcat'], "Unknown act_type"
#         assert conv_mode in ['learned', 'fixed', 'dynamic'], "Unknown conv_mode"
#         self.act_type = act_type
#         with self.name_scope():
#             self.features = nn.HybridSequential(prefix='')
#             if addstem:
#                 self.features.add(nn.Conv2D(channels=channels, kernel_size=3, strides=2,
#                                             padding=1, use_bias=False))
#                 self.features.add(nn.BatchNorm(in_channels=channels))
#                 self.features.add(nn.Activation('relu'))
#                 self.features.add(nn.Conv2D(channels=channels, kernel_size=3, strides=1,
#                                             padding=1, use_bias=False))
#                 self.features.add(nn.BatchNorm(in_channels=channels))
#                 self.features.add(nn.Activation('relu'))
#                 self.features.add(nn.Conv2D(channels=channels*2, kernel_size=3, strides=1,
#                                          padding=1, use_bias=False))
#                 self.features.add(nn.BatchNorm(in_channels=channels*2))
#                 self.features.add(nn.Activation('relu'))
#                 self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
#
#             for i, dilation in enumerate(dilations[:-1]):
#                 self.features.add(self._make_layer(
#                     dilation=dilation, channels=channels, stage_index=i, conv_mode=conv_mode,
#                     act_type=act_type, skernel=skernel, act_dilation=act_dilation,
#                     useReLU=useReLU, check_fullly=check_fullly, act_layers=act_layers,
#                     act_order=act_order, asBackbone=asBackbone))
#
#             self.features.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilations[-1],
#                                         padding=dilations[-1]))
#             self.features.add(nn.BatchNorm())
#
#             self.attention = nn.HybridSequential(prefix='')
#             if act_type == 'MSSeqATACConcat':
#                 self.attention.add(MSSeqAttentionMap(
#                     skernel=skernel, channels=channels, dilation=act_dilation,
#                     useReLU=useReLU, asBackbone=asBackbone))
#             elif act_type == 'xUnit':
#                 self.attention.add(xUnitAttentionMap(channels=channels, skernel_size=5))
#             elif act_type == 'relu':
#                 self.attention.add(nn.Activation('relu'))
#
#             if use_act_head:
#                 self.head = ATAC_FCNHead(head_act=act_type, useReLU=useReLU,
#                                          in_channels=channels, channels=classes)
#             else:
#                 self.head = _FCNHead(in_channels=channels, channels=classes)
#
#     def _make_layer(self, dilation, channels, stage_index, conv_mode, act_type, skernel,
#                     act_dilation, useReLU, check_fullly, act_layers, act_order, asBackbone):
#         layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
#         with layer.name_scope():
#
#             if check_fullly:
#                 if act_order == 'bac':
#                     # 后面的层优先用 Attention
#                     if stage_index + act_layers < 5:
#                         act_type = 'relu'
#                 elif act_order == 'pre':
#                  # 前面的层优先用 Attention
#                     if act_layers - stage_index - 1 < 0:
#                         act_type = 'relu'
#                 else:
#                     raise ValueError('Unknown act_order')
#
#             if conv_mode == 'fixed':
#
#                 layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
#                                     padding=dilation))
#                 layer.add(nn.BatchNorm())
#
#                 if act_type == 'prelu':
#                     layer.add(nn.PReLU())
#                 elif act_type == 'relu':
#                     layer.add(nn.Activation('relu'))
#                 elif act_type == 'swish':
#                     layer.add(nn.Swish())
#                 elif act_type == 'xUnit':
#                     layer.add(xUnit(channels=channels, skernel_size=5))
#                 elif act_type == 'SpaATAC':
#                     layer.add(SpaATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                       useReLU=useReLU, asBackbone=asBackbone))
#                 elif act_type == 'ChaATAC':
#                     layer.add(ChaATAC(channels=channels, useReLU=useReLU, useGlobal=False,
#                                       asBackbone=asBackbone))
#                 elif act_type == 'SeqATAC':
#                     layer.add(SeqATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                       useReLU=useReLU, asBackbone=asBackbone))
#                     # layer.add(DilatedSeqATACBackbone(channels=channels, dilation=act_dilation))
#                 elif act_type == 'MSSeqATAC':
#                     layer.add(MSSeqATAC(skernel=skernel, channels=channels, dilation=act_dilation,
#                                         useReLU=useReLU, asBackbone=asBackbone))
#                     # layer.add(DilatedSeqATACBackbone(channels=channels, dilation=act_dilation))
#                 elif act_type == 'MSSeqATACAdd':
#                     layer.add(MSSeqATACAdd(skernel=skernel, channels=channels,
#                                            dilation=act_dilation, useReLU=useReLU,
#                                            asBackbone=asBackbone))
#                 elif act_type == 'MSSeqATACConcat':
#                     layer.add(MSSeqATACConcat(skernel=skernel, channels=channels,
#                                               dilation=act_dilation, useReLU=useReLU,
#                                               asBackbone=asBackbone))
#                 else:
#                     raise ValueError('Unknown act_type')
#
#             elif conv_mode == 'learned':
#                 layer.add(LearnedCell(channels=channels, dilations=dilation))
#             elif conv_mode == 'dynamic':
#                 layer.add(DynamicCell(channels=channels, dilations=dilation))
#             else:
#                 raise ValueError('Unknown conv_mode')
#         return layer
#
#     def hybrid_forward(self, F, x):
#
#         _, _, hei, wid = x.shape
#         x = self.features(x)
#         if self.act_type == 'relu':
#             x = self.attention(x)
#         elif self.act_type == 'MSSeqATACConcat' or self.act_type == 'xUnit':
#             a = self.attention(x)
#             x = x * a
#         # elif self.act_type == 'xUnit':
#         #     a = self.attention(x)
#         #     x = x * a
#         else:
#             raise ValueError("Unknown self.act_type")
#
#         x = self.head(x)
#
#         out = F.contrib.BilinearResize2D(x, height=hei, width=wid)
#
#         return out
#
#     def evaluate(self, x):
#         """evaluating network with inputs and targets"""
#         return self.forward(x)


# class BasicContextFPN(HybridBlock):
#     def __init__(self, dilations=[1, 1, 2, 4, 8, 16], channels=16, classes=1,
#                  conv_mode='xxx', fuse_mode='xxx', act_type='relu', skernel=3, act_dilation=16,
#                  useReLU=False, use_act_head=False, check_fullly=False, act_layers=4,
#                  act_order='xxx', asBackbone=False, addstem=False, maxpool=True, **kwargs):
#         super(BasicContextFPN, self).__init__(**kwargs)
#
#         assert act_type in ['swish', 'prelu', 'relu', 'xUnit', 'SeqATAC', 'SpaATAC', 'ChaATAC',
#                             'MSSeqATAC', 'MSSeqATACAdd', 'MSSeqATACConcat'], "Unknown act_type"
#         assert conv_mode in ['fixed', 'learned', 'ChaDyReF', 'SeqDyReF', 'SK_ChaDyReF',
#                              'SK_1x1DepthDyReF', 'SK_MSSpaDyReF', 'SK_SpaDyReF', 'Direct_Add',
#                              'SKCell', 'SK_SeqDyReF', 'Sub_MSSpaDyReF', 'SK_MSSeqDyReF'], \
#             "Unknown conv_mode"
#         # assert fuse_mode in ['Direct_Add', 'SK', 'SK_MSSpa', 'LocalCha', 'GlobalCha', 'LocalGlobalCha', 'MSSpaLGCha'], \
#         #     "Unknown fuse_mode"
#         stem_width = int(channels // 2)
#         self.layer_num = len(dilations)
#         with self.name_scope():
#             self.stem = nn.HybridSequential(prefix='stem')
#             if addstem:
#                 self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
#                                             padding=1, use_bias=False))
#                 self.stem.add(nn.BatchNorm(in_channels=stem_width))
#                 self.stem.add(nn.Activation('relu'))
#                 self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
#                                             padding=1, use_bias=False))
#                 self.stem.add(nn.BatchNorm(in_channels=stem_width))
#                 self.stem.add(nn.Activation('relu'))
#                 self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
#                                          padding=1, use_bias=False))
#                 self.stem.add(nn.BatchNorm(in_channels=stem_width*2))
#                 self.stem.add(nn.Activation('relu'))
#             if maxpool:
#                 self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
#
#             self.stage_1 = nn.HybridSequential(prefix='stage_1')
#             self.stage_1.add(self._make_layer(
#                 dilation=dilations[0], channels=channels, stage_index=0, conv_mode=conv_mode,
#                 act_type=act_type, skernel=skernel, act_dilation=act_dilation,
#                 useReLU=useReLU, check_fullly=check_fullly, act_layers=act_layers,
#                 act_order=act_order, asBackbone=asBackbone))
#             if self.layer_num >= 2:
#                 self.stage_1.add(self._make_layer(
#                     dilation=dilations[1], channels=channels, stage_index=1, conv_mode=conv_mode,
#                     act_type=act_type, skernel=skernel, act_dilation=act_dilation,
#                     useReLU=useReLU, check_fullly=check_fullly, act_layers=act_layers,
#                     act_order=act_order, asBackbone=asBackbone))
#
#             # (1, 1, 2)
#             if self.layer_num >= 3:
#                 self.stage_2 = self._make_layer(
#                     dilation=dilations[2], channels=channels, stage_index=2,
#                     conv_mode=conv_mode, act_type=act_type, skernel=skernel,
#                     act_dilation=act_dilation, useReLU=useReLU, check_fullly=check_fullly,
#                     act_layers=act_layers, act_order=act_order, asBackbone=asBackbone)
#                 self.fuse12 = self._fuse_layer(fuse_mode=fuse_mode, channels=channels,
#                                                act_dilation=act_dilation, useReLU=useReLU,
#                                                fuse_index=12)
#
#             # (1, 1, 2, 4)
#             if self.layer_num >= 4:
#                 self.stage_3 = self._make_layer(
#                     dilation=dilations[3], channels=channels, stage_index=3,
#                     conv_mode=conv_mode, act_type=act_type, skernel=skernel,
#                     act_dilation=act_dilation, useReLU=useReLU, check_fullly=check_fullly,
#                     act_layers=act_layers, act_order=act_order, asBackbone=asBackbone)
#                 self.fuse23 = self._fuse_layer(fuse_mode=fuse_mode, channels=channels,
#                                                act_dilation=act_dilation, useReLU=useReLU,
#                                                fuse_index=23)
#
#             # (1, 1, 2, 4, 8)
#             if self.layer_num >= 5:
#                 self.stage_4 = self._make_layer(
#                     dilation=dilations[4], channels=channels, stage_index=4,
#                     conv_mode=conv_mode, act_type=act_type, skernel=skernel,
#                     act_dilation=act_dilation, useReLU=useReLU, check_fullly=check_fullly,
#                     act_layers=act_layers, act_order=act_order, asBackbone=asBackbone)
#                 self.fuse34 = self._fuse_layer(fuse_mode=fuse_mode, channels=channels,
#                                                act_dilation=act_dilation, useReLU=useReLU,
#                                                fuse_index=34)
#
#             # (1, 1, 2, 4, 8, 16)
#             if self.layer_num >= 6:
#                 self.stage_5 = self._make_layer(
#                     dilation=dilations[5], channels=channels, stage_index=5,
#                     conv_mode=conv_mode, act_type=act_type, skernel=skernel,
#                     act_dilation=act_dilation, useReLU=useReLU, check_fullly=check_fullly,
#                     act_layers=act_layers, act_order=act_order, asBackbone=asBackbone)
#                 self.fuse45 = self._fuse_layer(fuse_mode=fuse_mode, channels=channels,
#                                                act_dilation=act_dilation, useReLU=useReLU,
#                                                fuse_index=45)
#
#             self.head = _FCNHead(in_channels=channels, channels=classes)
#
#     def _make_layer(self, dilation, channels, stage_index, conv_mode, act_type, skernel,
#                     act_dilation, useReLU, check_fullly, act_layers, act_order, asBackbone):
#         layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
#         with layer.name_scope():
#
#             if conv_mode == 'fixed':
#                 layer.add(nn.Conv2D(channels=channels, kernel_size=3, dilation=dilation,
#                                     padding=dilation))
#             elif conv_mode == 'learned':
#                 layer.add(LearnedConv(channels=channels, dilations=dilation))
#             elif conv_mode == 'ChaDyReF':
#                 layer.add(ChaDyReFConv(channels=channels, dilations=dilation))
#             elif conv_mode == 'SK_ChaDyReF':
#                 layer.add(SK_ChaDyReFConv(channels=channels, dilations=dilation))
#             elif conv_mode == 'SK_1x1DepthDyReF':
#                 layer.add(SK_1x1DepthDyReFConv(channels=channels, dilations=dilation))
#             elif conv_mode == 'SK_MSSpaDyReF':
#                 layer.add(SK_MSSpaDyReFConv(channels=channels, dilations=dilation,
#                                             asBackbone=asBackbone))
#             elif conv_mode == 'Direct_Add':
#                 layer.add(Direct_AddConv(channels=channels, dilations=dilation,
#                                             asBackbone=asBackbone))
#             elif conv_mode == 'SK_SpaDyReF':
#                 layer.add(SK_SpaDyReFConv(channels=channels, dilations=dilation,
#                                           act_dilation=act_dilation))
#             elif conv_mode == 'SKCell':
#                 layer.add(SKConv(channels=channels, dilations=dilation))
#             elif conv_mode == 'SeqDyReF':
#                 layer.add(SeqDyReFConv(channels=channels, dilations=dilation,
#                                        act_dilation=act_dilation, useReLU=useReLU,
#                                        asBackbone=asBackbone))
#             elif conv_mode == 'SK_SeqDyReF':
#                 layer.add(SK_SeqDyReFConv(channels=channels, dilations=dilation,
#                                        act_dilation=act_dilation, useReLU=useReLU,
#                                        asBackbone=asBackbone))
#             else:
#                 raise ValueError('Unknown conv_mode')
#
#             layer.add(nn.BatchNorm())
#             layer.add(nn.Activation('relu'))
#
#         return layer
#
#     def _fuse_layer(self, fuse_mode, channels, act_dilation, useReLU, fuse_index):
#         # fuse_layer = nn.HybridSequential(prefix='fuse%d_' % fuse_index)
#
#         if fuse_mode == 'Direct_Add':
#             # fuse_layer.add(Direct_AddFuse(channels=channels))
#             fuse_layer = Direct_AddFuse(channels=channels)
#         elif fuse_mode == 'SK':
#             fuse_layer = SKFuse(channels=channels)
#         elif fuse_mode == 'LocalCha':
#             fuse_layer = LocalChaFuse(channels=channels)
#         elif fuse_mode == 'GlobalCha':
#             fuse_layer = GlobalChaFuse(channels=channels)
#         elif fuse_mode == 'LocalGlobalCha':
#             fuse_layer = LocalGlobalChaFuse(channels=channels)
#         elif fuse_mode == 'LocalSpa':
#             fuse_layer = LocalSpaFuse(channels=channels, act_dilation=act_dilation)
#         elif fuse_mode == 'GlobalSpa':
#             fuse_layer = GlobalSpaFuse(channels=channels, act_dilation=act_dilation)
#         elif fuse_mode == 'SK_MSSpa':
#             # fuse_layer.add(SK_MSSpaFuse(channels=channels, act_dilation=act_dilation))
#             fuse_layer = SK_MSSpaFuse(channels=channels, act_dilation=act_dilation)
#         else:
#             raise ValueError('Unknown fuse_mode')
#
#         return fuse_layer
#
#     def hybrid_forward(self, F, x):
#
#         _, _, hei, wid = x.shape
#
#         xs = self.stem(x)      # Subsampling 4
#         x1 = self.stage_1(xs)  # Subsampling 4, dilation 1
#
#         if self.layer_num <= 2:
#             xf = x1
#         elif self.layer_num == 3:
#             x2 = self.stage_2(x1)  # Subsampling 4, dilation 2
#             xf = self.fuse12(x2, x1)
#             # xf = x2 + x1
#         elif self.layer_num == 4:
#             x2 = self.stage_2(x1)  # Subsampling 4, dilation 2
#             x3 = self.stage_3(x2)  # Subsampling 4, dilation 4
#             xf = self.fuse23(x3, x2)
#             xf = self.fuse12(xf, x1)
#             # xf = x3 + x2
#             # xf = xf + x1
#         elif self.layer_num == 5:
#             x2 = self.stage_2(x1)  # Subsampling 4, dilation 2
#             x3 = self.stage_3(x2)  # Subsampling 4, dilation 4
#             x4 = self.stage_4(x3)  # Subsampling 4, dilation 8
#             xf = self.fuse34(x4, x3)
#             xf = self.fuse23(xf, x2)
#             xf = self.fuse12(xf, x1)
#             # xf = x4 + x3
#             # xf = xf + x2
#             # xf = xf + x1
#         elif self.layer_num == 6:
#             x2 = self.stage_2(x1)  # Subsampling 4, dilation 2
#             x3 = self.stage_3(x2)  # Subsampling 4, dilation 4
#             x4 = self.stage_4(x3)  # Subsampling 4, dilation 8
#             x5 = self.stage_5(x4)  # Subsampling 4, dilation 16
#             xf = self.fuse45(x5, x4)
#             xf = self.fuse34(xf, x3)
#             xf = self.fuse23(xf, x2)
#             xf = self.fuse12(xf, x1)
#             # xf = x5 + x4
#             # xf = xf + x3
#             # xf = xf + x2
#             # xf = xf + x1
#
#         xo = self.head(xf)
#
#         out = F.contrib.BilinearResize2D(xo, height=hei, width=wid)
#
#         return out
#
#     def evaluate(self, x):
#         """evaluating network with inputs and targets"""
#         return self.forward(x)



# class MPCMResNetFPN(HybridBlock):
#     def __init__(self, layers, channels, shift=3, classes=1,
#                  norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
#         super(MPCMResNetFPN, self).__init__(**kwargs)
#
#         self.layer_num = len(layers)
#         with self.name_scope():
#
#             self.shift = shift
#
#             stem_width = int(channels[0])
#             self.stem = nn.HybridSequential(prefix='stem')
#             self.stem.add(norm_layer(scale=False, center=False,
#                                      **({} if norm_kwargs is None else norm_kwargs)))
#
#             self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
#                                      padding=1, use_bias=False))
#             self.stem.add(norm_layer(in_channels=stem_width))
#             self.stem.add(nn.Activation('relu'))
#             self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
#                                      padding=1, use_bias=False))
#             self.stem.add(norm_layer(in_channels=stem_width))
#             self.stem.add(nn.Activation('relu'))
#             self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
#                                      padding=1, use_bias=False))
#             self.stem.add(norm_layer(in_channels=stem_width*2))
#             self.stem.add(nn.Activation('relu'))
#
#             self.head = _FCNHead(in_channels=channels[-1], channels=classes)
#
#             self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
#                                            channels=channels[1], stride=1, stage_index=1,
#                                            in_channels=channels[1])
#
#             self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
#                                            channels=channels[2], stride=2, stage_index=2,
#                                            in_channels=channels[1])
#
#             self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[2],
#                                            channels=channels[3], stride=2, stage_index=3,
#                                            in_channels=channels[2])
#
#             self.inc_c2 = nn.HybridSequential(prefix='inc_c2')
#             self.inc_c2.add(nn.Conv2D(channels=channels[3], kernel_size=1, strides=1,
#                                      padding=0, use_bias=False))
#             self.inc_c2.add(norm_layer(in_channels=channels[-1]))
#             self.inc_c2.add(nn.Activation('relu'))
#
#             self.inc_c1 = nn.HybridSequential(prefix='inc_c1')
#             self.inc_c1.add(nn.Conv2D(channels=channels[3], kernel_size=1, strides=1,
#                                      padding=0, use_bias=False))
#             self.inc_c1.add(norm_layer(in_channels=channels[-1]))
#             self.inc_c1.add(nn.Activation('relu'))
#
#
#     def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
#                     norm_layer=BatchNorm, norm_kwargs=None):
#         layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
#         with layer.name_scope():
#             downsample = (channels != in_channels) or (stride != 1)
#             layer.add(block(channels, stride, downsample, in_channels=in_channels,
#                             prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
#             for _ in range(layers-1):
#                 layer.add(block(channels, 1, False, in_channels=channels, prefix='',
#                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))
#         return layer
#
#     def hybrid_forward(self, F, x):
#
#         _, _, orig_hei, orig_wid = x.shape
#         x = self.stem(x)      # sub 2
#         c1 = self.layer1(x)   # sub 2
#         _, _, c1_hei, c1_wid = c1.shape
#         c2 = self.layer2(c1)  # sub 4
#         _, _, c2_hei, c2_wid = c2.shape
#         c3 = self.layer3(c2)  # sub 8
#         _, _, c3_hei, c3_wid = c3.shape
#
#         # 1. upsampling(c3) -> c3PCM   # size: sub 4
#
#         # c3 -> c3PCM
#         # 2. pwconv(c2) -> c2PCM       # size: sub 4
#         # 3. upsampling(c3PCM + c2PCM) # size: sub 2
#         # 4. pwconv(c1) -> c1PCM       # size: sub 2
#         # 5. upsampling(upsampling(c3PCM + c2PCM)) + c1PCM
#         # 6. upsampling(upsampling(c3PCM + c2PCM)) + c1PCM
#
#         c3pcm = self.cal_pcm(c3, shift=self.shift)
#         up_c3pcm = F.contrib.BilinearResize2D(c3pcm, height=c2_hei, width=c2_wid) # sub 4, 64
#
#         inc_c2 = self.inc_c2(c2)               # sub 4, 64
#         c2pcm = self.cal_pcm(inc_c2, shift=self.shift)
#
#         c23pcm = up_c3pcm + c2pcm              # sub 4, 64
#
#         up_c23pcm = F.contrib.BilinearResize2D(c23pcm, height=c1_hei, width=c1_wid)  # sub 2, 64
#         inc_c1 = self.inc_c1(c1)               # sub 2, 64
#         c1pcm = self.cal_pcm(inc_c1, shift=self.shift)
#
#         out = up_c23pcm + c1pcm              # sub 2, 64
#         pred = self.head(out)
#         out = F.contrib.BilinearResize2D(pred, height=orig_hei, width=orig_wid)
#
#         return out
#
#     def evaluate(self, x):
#         """evaluating network with inputs and targets"""
#         return self.forward(x)
#
#     def circ_shift(self, cen, shift):
#
#         _, _, hei, wid = cen.shape
#
#         ######## B1 #########
#         # old: AD  =>  new: CB
#         #      BC  =>       DA
#         B1_NW = cen[:, :, shift:, shift:]          # B1_NW is cen's SE
#         B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
#         B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
#         B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
#         B1_N = nd.concat(B1_NW, B1_NE, dim=3)
#         B1_S = nd.concat(B1_SW, B1_SE, dim=3)
#         B1 = nd.concat(B1_N, B1_S, dim=2)
#
#         ######## B2 #########
#         # old: A  =>  new: B
#         #      B  =>       A
#         B2_N = cen[:, :, shift:, :]          # B2_N is cen's S
#         B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
#         B2 = nd.concat(B2_N, B2_S, dim=2)
#
#         ######## B3 #########
#         # old: AD  =>  new: CB
#         #      BC  =>       DA
#         B3_NW = cen[:, :, shift:, wid-shift:]          # B3_NW is cen's SE
#         B3_NE = cen[:, :, shift:, :wid-shift]      # B3_NE is cen's SW
#         B3_SW = cen[:, :, :shift, wid-shift:]      # B3_SW is cen's NE
#         B3_SE = cen[:, :, :shift, :wid-shift]          # B1_SE is cen's NW
#         B3_N = nd.concat(B3_NW, B3_NE, dim=3)
#         B3_S = nd.concat(B3_SW, B3_SE, dim=3)
#         B3 = nd.concat(B3_N, B3_S, dim=2)
#
#         ######## B4 #########
#         # old: AB  =>  new: BA
#         B4_W = cen[:, :, :, wid-shift:]          # B2_W is cen's E
#         B4_E = cen[:, :, :, :wid-shift]          # B2_E is cen's S
#         B4 = nd.concat(B4_W, B4_E, dim=3)
#
#         ######## B5 #########
#         # old: AD  =>  new: CB
#         #      BC  =>       DA
#         B5_NW = cen[:, :, hei-shift:, wid-shift:]          # B5_NW is cen's SE
#         B5_NE = cen[:, :, hei-shift:, :wid-shift]      # B5_NE is cen's SW
#         B5_SW = cen[:, :, :hei-shift, wid-shift:]      # B5_SW is cen's NE
#         B5_SE = cen[:, :, :hei-shift, :wid-shift]          # B5_SE is cen's NW
#         B5_N = nd.concat(B5_NW, B5_NE, dim=3)
#         B5_S = nd.concat(B5_SW, B5_SE, dim=3)
#         B5 = nd.concat(B5_N, B5_S, dim=2)
#
#         ######## B6 #########
#         # old: A  =>  new: B
#         #      B  =>       A
#         B6_N = cen[:, :, hei-shift:, :]          # B6_N is cen's S
#         B6_S = cen[:, :, :hei-shift, :]      # B6_S is cen's N
#         B6 = nd.concat(B6_N, B6_S, dim=2)
#
#         ######## B7 #########
#         # old: AD  =>  new: CB
#         #      BC  =>       DA
#         B7_NW = cen[:, :, hei-shift:, shift:]          # B7_NW is cen's SE
#         B7_NE = cen[:, :, hei-shift:, :shift]      # B7_NE is cen's SW
#         B7_SW = cen[:, :, :hei-shift, shift:]      # B7_SW is cen's NE
#         B7_SE = cen[:, :, :hei-shift, :shift]          # B7_SE is cen's NW
#         B7_N = nd.concat(B7_NW, B7_NE, dim=3)
#         B7_S = nd.concat(B7_SW, B7_SE, dim=3)
#         B7 = nd.concat(B7_N, B7_S, dim=2)
#
#         ######## B8 #########
#         # old: AB  =>  new: BA
#         B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
#         B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
#         B8 = nd.concat(B8_W, B8_E, dim=3)
#
#         return B1, B2, B3, B4, B5, B6, B7, B8
#
#     def cal_pcm(self, cen, shift):
#
#         B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=shift)
#         s1 = (B1 - cen) * (B5 - cen)
#         s2 = (B2 - cen) * (B6 - cen)
#         s3 = (B3 - cen) * (B7 - cen)
#         s4 = (B4 - cen) * (B8 - cen)
#
#         c12 = nd.minimum(s1, s2)
#         c123 = nd.minimum(c12, s3)
#         c1234 = nd.minimum(c123, s4)
#
#         return c1234


