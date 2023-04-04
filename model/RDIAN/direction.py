import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Conv_d11(nn.Module):
    def __init__(self):            
        super(Conv_d11, self).__init__()
        kernel = [[-1, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2) 
        
class Conv_d12(nn.Module):
    def __init__(self):            
        super(Conv_d12, self).__init__()
        kernel = [[0, 0, -1, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)    


class Conv_d13(nn.Module):
    def __init__(self):            
        super(Conv_d13, self).__init__()
        kernel = [[0, 0, 0, 0, -1],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)    


class Conv_d14(nn.Module):
    def __init__(self):            
        super(Conv_d14, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,-1],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)    


class Conv_d15(nn.Module):
    def __init__(self):            
        super(Conv_d15, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,-1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)         
        
class Conv_d16(nn.Module):
    def __init__(self):            
        super(Conv_d16, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,-1,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)   

class Conv_d17(nn.Module):
    def __init__(self):            
        super(Conv_d17, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [-1,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)         
        
class Conv_d18(nn.Module):
    def __init__(self):            
        super(Conv_d18, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [-1, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)         