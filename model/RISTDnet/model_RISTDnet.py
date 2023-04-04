from .RISTDnet_parts import *
import torch.nn as nn

class RISTDnet(nn.Module):
    def __init__(self):
        super(RISTDnet, self).__init__()
        self.FW = FENetwFW()
        self.FV = FENetwVW()
        self.FM = FMNet()

    def forward(self, img):
        FW_out = self.FW.forward(img)
        FV_out = self.FV(FW_out)
        FM_out = self.FM(FV_out, img.shape[0], img.shape[2], img.shape[3])
        return FM_out.sigmoid()
