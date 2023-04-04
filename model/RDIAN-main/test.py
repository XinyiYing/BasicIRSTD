import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
import visdom

from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import math
import numpy as np
import time
import re
import scipy.io

from utils.data import IRDSTDataset
from model.metrics import SigmoidMetric, SamplewiseSigmoidMetric, ROCMetric, T_ROCMetric
from model.segmentation import RDIAN

device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def parse_args():
    # Setting parameters
    parser = ArgumentParser(description='Implement of RDIAN model')
    # Size of images
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')

    args = parser.parse_args()
    return args

class test(object):
    def __init__(self, args):
        self.args = args

        self.unloader = transforms.ToPILImage()

        ## dataset
        testset = IRDSTDataset(args, mode='val')
        self.test_data_loader = Data.DataLoader(testset, batch_size=1, num_workers=8, shuffle = False)

        ## model
        self.net = RDIAN()        
        self.net.load_state_dict(torch.load('params/best.pth', map_location=lambda storage, loc: storage))         
        self.net.eval()
        self.net = self.net.cuda()

        ## folders
        folder_name = '%s_RDIAN' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),)
        self.save_folder = ops.join('testresult/', folder_name)
        self.save_img = ops.join(self.save_folder, 'image')
        if not ops.exists('resultmydata'):
            os.mkdir('resultmydata')
        if not ops.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not ops.exists(self.save_img):
            os.mkdir(self.save_img)
            
        ## evaluation metrics
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.roc = ROCMetric(1,bins=255)
        # self.t_roc = T_ROCMetric(1,bins=255)
        self.best_iou = 0
        self.best_nIoU = 0            

    def testing(self):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        
        tbar = tqdm(self.test_data_loader)
        self.names = []
        base_dir = 'datasets/example/'
        txtfile = 'test.txt'
        self.list_dir = ops.join(base_dir,  txtfile)
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]       
            
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.net(data.cuda())
                output1 = output.cpu()
                output = F.sigmoid(output)
                
                img = output.cpu().clone()
                img = img.squeeze(0)
                img = self.unloader(img)
                # save result
                name = self.names[i]
                img_name =  name +'.bmp'
                img.save(ops.join(self.save_img, img_name))

            self.iou_metric.update(output1, labels)
            self.nIoU_metric.update(output1, labels)
            
            pixAcc, IoU = self.iou_metric.get()
            _, nIoU,DR = self.nIoU_metric.get()           
            
            tbar.set_description('  pixAcc:%f, IoU:%f, nIoU:%f, DR:%f'
                                 %( pixAcc, IoU, nIoU,DR))                            

if __name__ == '__main__':
    args = parse_args()

    test = test(args)
    test.testing()
    print("over")


