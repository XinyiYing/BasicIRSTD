import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ISNet.my_functionals import GatedSpatialConv as gsc
from model.ISNet.network import Resnet
from torch.nn.parameter import Parameter
from model.ISNet.DCNv2.TTOA import TTOA
from utils import *

'''
ISNet_TTOA
'''

class TFD(nn.Module):
    def __init__(self, inch, outch):
        super(TFD, self).__init__()
        self.res1 = Resnet.BasicBlock1(inch, outch, stride=1, downsample=None)
        self.res2 = Resnet.BasicBlock1(inch, outch, stride=1, downsample=None)
        self.gate = gsc.GatedSpatialConv2d(inch, outch)
    def forward(self,x,f_x):
        u_0 = x
        u_1, delta_u_0 = self.res1(u_0)
        _, u_2 = self.res2(u_1)
        u_3_pre = self.gate(u_2, f_x)
        u_3 = 3 * delta_u_0 + u_2 + u_3_pre
        return u_3

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
    
class ISNet(nn.Module):
    def __init__(self, mode='test', layer_blocks=[4] * 3, channels=[8, 16, 32, 64]):
        super(ISNet, self).__init__()

        self.mode = mode
        stem_width = int(channels[0])
        self.stem = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, 2*stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*stem_width),
            nn.ReLU(True),

            nn.MaxPool2d(3, 2, 1),
        )
        self.TTOA_low = TTOA(channels[2],channels[2])
        self.TTOA_high = TTOA(channels[1],channels[1])
        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)

        self.uplayer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                         in_channels=channels[2], out_channels=channels[2], stride=1)

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)

        self.uplayer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                         in_channels=channels[1], out_channels=channels[1], stride=1)

        self.head = _FCNHead(channels[1], 1)
        #edge branch
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(32, 1, 1)
        self.dsn3 = nn.Conv2d(16, 1, 1)

        self.res1 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)

        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)

        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
        self.sigmoid = nn.Sigmoid()
        self.SA = sa_layer(4, 2)
        self.SA_att = sa_layer(17, 2)
        self.dsup = nn.Conv2d(1, 64, 1)
        self.head2 = _FCNHead(channels[1], 3)
        self.conv2_1 = nn.Conv2d(3, 1, 1)
        self.conv16 = nn.Conv2d(3, 16, 1)
        self.myb1 = TFD(64,64)
        self.myb2 = TFD(64,64)
        self.myb3 = TFD(64,64)
        
        self.grad = Get_gradient_nopadding()

    def forward(self, x):
        x_grad = self.grad(x)
        _, _, hei, wid = x.shape
        x_size = x.size()

        x1 = self.stem(x)
        c1 = self.layer1(x1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)



        deconc2 = self.deconv2(c3)
        fusec2 = self.TTOA_low(deconc2, c2)
        upc2 = self.uplayer2(fusec2)

        deconc1 = self.deconv1(upc2)
        fusec1 = self.TTOA_high(deconc1, c1)
        upc1 = self.uplayer1(fusec1)

        s1 = F.interpolate(self.dsn1(c3), size=[hei, wid ], mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.dsn2(upc2), size=[hei , wid], mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.dsn3(upc1), size=[hei, wid], mode='bilinear', align_corners=True)

        m1f = F.interpolate(x_grad, size=[hei, wid], mode='bilinear', align_corners=True)
        m1f = self.dsup(m1f)
        cs1 = self.myb1(m1f, s1)
        cs2 = self.myb2(cs1, s2)
        cs = self.myb3(cs2, s3)
        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)

        # cat = torch.cat((edge_out, x_grad), dim=1)
        # cat = self.SA(cat)
        # acts = self.cw(cat)
        # acts = self.sigmoid(acts)
        upc1 = F.interpolate(upc1, size=[hei, wid], mode='bilinear')
        fuse = edge_out * upc1 + upc1



        pred = self.head(fuse)

        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')
        if self.mode == 'train':
            return self.sigmoid(out), edge_out
        else:
            return self.sigmoid(out)

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0

if __name__ == '__main__':
    net = ISNet(layer_blocks = [4] * 3,
        channels = [8, 16, 32, 64])
    print(net)