import torch
import torch.nn as nn

conv1_2 = nn.Conv2d(64, 64, (1,3), 1, (0,1))
conv1_1 = nn.Conv2d(64, 64, 1, 1, 0)
a = torch.randn(64,64,8,8)
b = conv1_2(a)
c = conv1_1(a)
d = b + c
print(d.shape)