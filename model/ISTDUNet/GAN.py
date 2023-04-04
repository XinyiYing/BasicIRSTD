import torch
import torch.nn as nn

from .resnet2020 import ResNet, Bottleneck, ResNetCt, ResNetDt

class Dis(nn.Module):
    def __init__(self,
                 inp_num = 1,
                 layers=[1, 2, 4, 8],
                 channels=[8, 16, 32, 64],
                 bottleneck_width=16,
                 stem_width=8,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 **kwargs
                 ):
        super(Dis, self).__init__()
        stemWidth = int(channels[0])
        self.stem = nn.Sequential(
            nn.Conv2d(2, stemWidth*2, kernel_size=3, stride=1, padding=1, bias=False),
            normLayer(stemWidth*2),
            activate()
        )
        self.down = ResNetDt(Bottleneck, layers, inp_num=inp_num,
                       radix=1, groups=1, bottleneck_width=bottleneck_width,
                       deep_stem=True, stem_width=stem_width, avg_down=True,
                       avd=True, avd_first=False, layer_parms=channels, **kwargs)

    def forward(self, x):
        # ret = []
        x = self.stem(x)
        # ret.append(x)
        x = self.down(x)
        ret = x
        return ret