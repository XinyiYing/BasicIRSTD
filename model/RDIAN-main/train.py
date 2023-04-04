import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
import visdom

from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import math
import numpy as np
import time
import random
import warnings

from utils.data import IRDSTDataset
from utils.data import DataLoaderX
from utils.data import DataPrefetcher
from utils.lr_scheduler import adjust_learning_rate
from model.segmentation import RDIAN
from model.loss import SoftLoULoss
from model.metrics import SigmoidMetric, SamplewiseSigmoidMetric
import pickle


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
    parser.add_argument('--use_cuda', type=str, default='True', help='use gpu ')
    parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 2,3. use -1 for CPU')
    parser.add_argument('--random_seed', type=str, default='42', help='0,1,.....')

    args = parser.parse_args()
    
    return args
    
def init_env(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

class Trainer(object):

    
    def __init__(self, args):
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        ## dataset
        trainset = IRDSTDataset(args, mode='train')
        valset = IRDSTDataset(args, mode='val')
        self.train_data_loader = DataLoaderX(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8,pin_memory=True,drop_last=True)
        self.val_data_loader = DataLoaderX(valset, batch_size=1, num_workers=8,pin_memory=True)

        ## model
        self.net = RDIAN()
        
        ## initialize
        self.net.apply(self.weight_init)
        if len(args.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net)
            print('----multi-GPU----')
        self.net = self.net.cuda()
        
        ## criterion
        self.criterion = SoftLoULoss()

        ## optimizer
        self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        ## evaluation metrics
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0

        ## folders
        folder_name = '%s_RDIAN' % (time.strftime('SBC%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
        self.save_folder = ops.join('resultmydata/', folder_name)
        self.save_pth = ops.join(self.save_folder, 'checkpoint')
        if not ops.exists('resultmydata'):
            os.mkdir('resultmydata')
        if not ops.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not ops.exists(self.save_pth):
            os.mkdir(self.save_pth)

        # ## SummaryWriter
        # self.writer = SummaryWriter(log_dir=self.save_folder)
        # self.writer.add_text(folder_name, 'Args:%s, ' % args)

        ## Print info
        print('folder: %s' % self.save_folder)
        print('Args: %s' % args)
        
    def training(self, epoch):
        # training step
        losses = []
        self.net.train() 
        self.iou_metric.reset()        
                
        tbar = tqdm(self.train_data_loader)
        for i, (data, masks) in enumerate(tbar):
            use_cuda = args.use_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            data = data.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            output = self.net(data)
            loss= self.criterion(output, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            self.log_name = os.path.join(self.save_folder, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('Epoch:%3d, lr:%f, train loss:%f'
                                 % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses)))
            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f'
                                 % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses)))
            
            # pixAcc, IoU = self.iou_metric.get()

        adjust_learning_rate(self.optimizer, epoch, args.epochs, args.learning_rate,
                             args.warm_up_epochs, 1e-6)

        

    def validation(self, epoch):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        eval_losses = []
        self.net.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, masks) in enumerate(tbar):
            with torch.no_grad():
                output = self.net(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, masks)
            eval_losses.append(loss.item())
            
            self.iou_metric.update(output, masks)
            self.nIoU_metric.update(output, masks)
            pixAcc, IoU = self.iou_metric.get()
            _, nIoU,DR = self.nIoU_metric.get()

            tbar.set_description('  Epoch:%3d, eval loss:%f, pixAcc:%f, IoU:%f, nIoU:%f,'# DR:%f'
                                 %(epoch, np.mean(eval_losses), pixAcc, IoU, nIoU))

        pth_name = 'Epoch-%3d_pixAcc-%4f_IoU-%.4f_nIoU-%.4f.pth' % (epoch, pixAcc, IoU, nIoU)
        if IoU > self.best_iou:
            torch.save(self.net.state_dict(), ops.join(self.save_pth, pth_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(self.net.state_dict(), ops.join(self.save_pth, pth_name))
            self.best_nIoU = nIoU

        # self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        # self.writer.add_scalar('Eval/IoU', IoU, epoch)
        # self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        # self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        # self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)


if __name__ == '__main__':
    args = parse_args()
    init_env(args.gpu_ids)

    trainer = Trainer(args)
    for epoch in range(1, args.epochs+1):
        trainer.training(epoch)
        trainer.validation(epoch)

    print('Best IoU: %.5f, best nIoU: %.5f' % (trainer.best_iou, trainer.best_nIoU))