import argparse
from net import Net
import os
import time
from thop import profile
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD Parameter and FLOPs")
parser.add_argument("--model_names", default=['ACM', 'ALCNet', 'DNANet', 'ISNet', 'RISTDnet', 'UIUNet', 'U-Net', 'RDIAN', 'ISTDU-Net'], nargs='+', 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'RISTDnet', 'UIUNet', 'U-Net', 'RDIAN', 'ISTDU-Net'")

global opt
opt = parser.parse_args()

if __name__ == '__main__':
    opt.f = open('./params_' + (time.ctime()).replace(' ', '_') + '.txt', 'w')
    input_img = torch.rand(1,1,256,256).cuda()
    for model_name in opt.model_names:
        net = Net(model_name, mode='test').cuda()    
        flops, params = profile(net, inputs=(input_img, ))
        print(model_name)
        print('Params: %2fM' % (params/1e6))
        print('FLOPs: %2fGFLOPs' % (flops/1e9))
        opt.f.write(model_name + '\n')
        opt.f.write('Params: %2fM\n' % (params/1e6))
        opt.f.write('FLOPs: %2fGFLOPs\n' % (flops/1e9))
        opt.f.write('\n')
    opt.f.close()
        
