import torch
import torch.nn as nn
from torch.nn import functional as F
from .CovKernelFW import get_kernels
from .FeatureMap import GenLikeMap
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

global global_step
global_step = 0

conv_writer = SummaryWriter(comment='--conv')


class FENetwFW(nn.Module):
    """
    基于固定权值卷积核的特征提取模块
    A feature extraction network based on convolution kernel with fixed weight.(FENetwFW)
    """

    def __init__(self):
        super(FENetwFW, self).__init__()
        kernels = [get_kernels(i) for i in range(1, 6)]  # 获取各种尺寸的卷积核
        self.weights = [
            nn.Parameter(data=torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
            for ks in kernels for k in ks
        ]  # 将卷积核转换为成pytorch中的格式

    def forward(self, img):
        feature_maps = [img]  # 融合的特征
        for ws in self.weights:  # 用各个卷积核对图像进行卷积，提取不同的特征图
            feature_maps.append(F.conv2d(img, ws, stride=1, padding = ws.shape[-1]//2))

        feature_maps = torch.cat(feature_maps, dim=1)  # 对各个卷积核卷积的结果进行融合

        conv_writer.add_image('固定权值卷积',
                              make_grid(torch.unsqueeze(feature_maps[0], dim=0).transpose(0, 1), normalize=True,
                                        nrow=3),
                              global_step=global_step)
        return feature_maps


class FENetwVW(nn.Module):
    """
    基于变化权值卷积核的特征提取模块
    A Feature extraction network based on convolution kernel with variable weight
    """

    def __init__(self):
        super(FENetwVW, self).__init__()
        self.c1 = nn.Conv2d(16, 32, kernel_size=(11, 11), padding=5, stride=(1, 1), bias=None)  # Convolution_1
        # torch.nn.init.xavier_normal_(self.c1.weight, gain=1.0)
        self.c2 = P1C2(32, 64)  # Pooling_1 & Convolution_2
        self.c3 = P2C3(64, 128)  # Pooling_2 & Convolution_3
        self.FCsubnet = FCsubnet(128, 256)  # Feature concatenation subnetwork
        self.c5 = nn.Conv2d(768, 128, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=None)  # Convolution_5

    def forward(self, fw_out):
        global global_step

        c1 = self.c1(fw_out)
        conv_writer.add_image('c1',
                              make_grid(torch.unsqueeze(c1[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
                              global_step=global_step)
        c2 = self.c2(c1)
        conv_writer.add_image('c2',
                              make_grid(torch.unsqueeze(c2[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
                              global_step=global_step)
        c3 = self.c3(c2)
        conv_writer.add_image('c3',
                              make_grid(torch.unsqueeze(c3[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
                              global_step=global_step)
        FC_subnet = self.FCsubnet(c3)
        conv_writer.add_image('特征级联子网络',
                              make_grid(torch.unsqueeze(FC_subnet[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
                              global_step=global_step)
        c5 = self.c5(FC_subnet)
        conv_writer.add_image('c5',
                              make_grid(torch.unsqueeze(c5[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
                              global_step=global_step)

        global_step += 1
        return c5


class FMNet(nn.Module):
    """
    特征图映射
    Feature mapping process
    """

    def __init__(self):
        super(FMNet, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, FV, batch_size, W, H):
        # first64 = FV[:, :64, :, :]  # 背景似然
        last64 = FV[:, 64:, :, :]  # 目标似然

        # bg_likemap = GenLikeMap(first64, batch_size, W, H)
        tg_likemap = GenLikeMap(last64, batch_size, W, H)
        # bg_likelihood = self.sigmoid(
        #     bg_likemap)

        # tg_likelihood = self.sigmoid(
        #     tg_likemap)
        return torch.unsqueeze(tg_likemap, dim=1)


class FCsubnet(nn.Module):
    """
    特征级联子网络
    A Feature concatenation subnetwork
    """

    def __init__(self, in_c, out_c):
        super(FCsubnet, self).__init__()
        self.reorg = ReOrg()  # 特征重组
        self.p3c4 = P3C4(in_c, out_c)  # Pooling_3 & Convolution_4

    def forward(self, c3):
        return torch.cat([self.reorg(c3), self.p3c4(c3)], dim=1)  # 特征拼接


class ReOrg(nn.Module):
    """
    特征重组
    """

    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, p2c3):
        w = p2c3.shape[2]
        h = p2c3.shape[3]
        pink = p2c3[:, :, :w // 2, :h // 2]
        green = p2c3[:, :, w // 2:, :h // 2]
        purple = p2c3[:, :, :w // 2, h // 2:]
        red = p2c3[:, :, w // 2:, h // 2:]
        return torch.cat([pink, green, purple, red], dim=1)


class P1C2(nn.Module):
    def __init__(self, in_c, out_c):
        super(P1C2, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=(7, 7), padding=3, stride=(1, 1), bias=None)

    def forward(self, c1):
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        return c2


class P2C3(nn.Module):
    def __init__(self, in_c, out_c):
        super(P2C3, self).__init__()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=(5, 5), padding=2, stride=(1, 1), bias=None)


    def forward(self, c2):
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        return c3


class P3C4(nn.Module):
    def __init__(self, in_c, out_c):
        super(P3C4, self).__init__()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=None)


    def forward(self, c3):
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        return c4
