import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import *
from .direction import *

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

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

class NewBlock(nn.Module):
    def __init__(self, in_channels, stride,kernel_size,padding):
        super(NewBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.layer2 = conv_batch(reduced_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out        

class RDIAN(nn.Module):
    def __init__(self):
    
        super(RDIAN, self).__init__()        
        accumulate_params = "none"
        self.conv1 = conv_batch(1, 16)
        self.conv2 = conv_batch(16, 32, stride=2)       
        self.residual_block0 = self.make_layer(NewBlock, in_channels=32, num_blocks=1, kernel_size=1,padding=0,stride=1)
        self.residual_block1 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=3,padding=1,stride=1)
        self.residual_block2 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=5,padding=2,stride=1)
        self.residual_block3 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=7,padding=3,stride=1)
        self.cbam  = CBAM(32, 32)        
        self.conv_cat = conv_batch(4*32, 32, 3, padding=1)
        self.conv_res = conv_batch(16, 32, 1, padding=0)
        self.relu = nn.ReLU(True)
        
        self.d11=Conv_d11()
        self.d12=Conv_d12()
        self.d13=Conv_d13()
        self.d14=Conv_d14()
        self.d15=Conv_d15()
        self.d16=Conv_d16()
        self.d17=Conv_d17()
        self.d18=Conv_d18()

        self.head = _FCNHead(32, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape
        d11 = self.d11(x)
        d12 = self.d12(x)
        d13 = self.d13(x)
        d14 = self.d14(x)
        d15 = self.d15(x)
        d16 = self.d16(x)
        d17 = self.d17(x)
        d18 = self.d18(x)
        md = d11.mul(d15) + d12.mul(d16) + d13.mul(d17) + d14.mul(d18)
        md = F.sigmoid(md)
        
        out1= self.conv1(x)        
        out2 = out1.mul(md)       
        out = self.conv2(out1 + out2)
            
        c0 = self.residual_block0(out)
        c1 = self.residual_block1(out)
        c2 = self.residual_block2(out)
        c3 = self.residual_block3(out)
 
        x_cat = self.conv_cat(torch.cat((c0, c1, c2, c3), dim=1)) #[16,32,240,240]
        x_a = self.cbam(x_cat)
        
        temp = F.interpolate(x_a, size=[hei, wid], mode='bilinear')
        temp2 = self.conv_res(out1)
        x_new = self.relu( temp + temp2)
        self.x_new = x_new
        pred = self.head(x_new)

        return pred.sigmoid()
             
    def make_layer(self, block, in_channels, num_blocks, stride, kernel_size, padding):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, stride, kernel_size, padding))
        return nn.Sequential(*layers)
