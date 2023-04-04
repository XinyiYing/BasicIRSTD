import torch
from torch import nn

from functools import reduce

class miNet(nn.Module):
    def __init__(self, inpNum=1):
        super(miNet, self).__init__()
        self.inpNum = inpNum
        # self.individual = self.funIndividual
        # self.pallet = self.funPallet
        # self.conbine = self.funConbine
        # self.encode = self.funEncode
        # self.decode = self.funDecode

    def funIndividual(self, x):
        return [x for _ in range(self.inpNum)]

    def funPallet(self, x):
        return x

    def funConbine(self, x):
        def add(x, y):
            return x+y
        return reduce(add, x)

    def funEncode(self, x):
        return x

    def funDecode(self, x):
        return x

    def funOutput(self,x):
        return x

    def forward(self, x):
        x = self.funIndividual(x)
        x = self.funPallet(x)
        x = self.funConbine(x)
        x = self.funEncode(x)
        x = self.funDecode(x)
        x = self.funOutput(x)
        return x

if __name__ == '__main__':
    net = miNet(2)
    x = torch.tensor([1,2,3])
    y = net([x,x])