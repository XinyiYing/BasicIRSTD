import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ACM'], nargs='+', 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'RISTDnet', 'UIUNet', 'U-Net', 'RDIAN', 'ISTDU-Net'")
parser.add_argument("--dataset_dir", default='../DNAnet/dataset/SIRST3', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'], nargs='+', 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--mask_pred_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

def eval(): 
    test_set = EvalSetLoader(opt.dataset_dir, opt.mask_pred_dir, opt.test_dataset_name, opt.model_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    for idx_iter, (pre_mask, gt_mask, size) in enumerate(test_loader):
        eval_mIoU.update(pre_mask>opt.threshold, gt_mask)
        eval_PD_FA.update(pre_mask[0,0,:,:]>opt.threshold, gt_mask[0,0,:,:], size)     
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open('./eval_' + (time.ctime()).replace(' ', '_') + '.txt', 'w')
    for model_name in opt.model_names:
        opt.model_name = model_name
        print(opt.model_name)
        opt.f.write(opt.model_name + '\n')
        for dataset_name in opt.dataset_names:
            opt.test_dataset_name = dataset_name
            print(opt.test_dataset_name)
            opt.f.write(opt.test_dataset_name + '\n')
            eval()
        print('\n')
        opt.f.write('\n')
    opt.f.close()
        
