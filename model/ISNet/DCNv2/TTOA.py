import torch
import torch.nn as nn
from model.ISNet.DCNv2.DCN.modules.deform_conv import DeformConvPack as DCN

class TTOA(nn.Module):
    def __init__(self,low_channels,high_channels,c_kernel=3,r_kernel=3,use_att=False,use_process=True):
        '''
                  :param low_channels: low_level feature channels
                  :param high_channels: high_level feature channels
                  :param c_kernel: colum dcn kernels kx1 just use k
                  :param r_kernel: row dcn kernels 1xk just use k
                  :param use_att: bools
                  :param use_process: bools
                  '''
        super(TTOA, self).__init__()

        self.l_c = low_channels
        self.h_c = high_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.att = use_att
        self.non_local_att = nn.Conv2d
        if self.l_c == self.h_c:
            print('Channel checked!')
        else:
            raise ValueError('Low and Hih channels need to be the same!')
        self.dcn_row = DCN(self.l_c,self.h_c,kernel_size=(1,self.r_k),stride=1,padding=(0,self.r_k//2))
        self.dcn_colum = DCN(self.l_c,self.h_c,kernel_size=(self.c_k,1),stride=1,padding=(self.c_k//2,0))
        self.sigmoid = nn.Sigmoid()
        if self.att == True:
            self.csa = self.non_local_att(self.l_c,self.h_c,1,1,0)
        else:
            self.csa = None
        if use_process == True:
            self.preprocess = nn.Sequential(nn.Conv2d(self.l_c,self.h_c//2,1,1,0),nn.Conv2d(self.h_c//2,self.l_c,1,1,0))
        else:
            self.preprocess = None
    def forward(self,a_low,a_high):
        if self.preprocess is not None:
            a_low = self.preprocess(a_low)
            a_high = self.preprocess(a_high)
        else:
            a_low = a_low
            a_high = a_high

        a_low_c = self.dcn_colum(a_low)
        a_low_cw = self.sigmoid(a_low_c)
        a_low_cw = a_low_cw * a_high
        a_colum = a_low + a_low_cw

        a_low_r = self.dcn_row(a_low)
        a_low_rw = self.sigmoid(a_low_r)
        a_low_rw = a_low_rw * a_high
        a_row = a_low + a_low_rw

        if self.csa is not None:
            a_TTOA = self.csa(a_row + a_colum)
        else:
            a_TTOA = a_row + a_colum
        return a_TTOA

#########Test ttoa
# img_low = torch.randn(32,3,512,512)
# img_high = torch.randn(32,3,512,512)
# a = TTOA(3,3)
# feature = a(img_low,img_high)
# print(feature.size())
#
