import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()
        self.unloader = transforms.ToPILImage()

    def IOU(self, pred, mask):
        smooth = 1

        intersection = pred * mask

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(mask, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
            (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        return loss

    '''
    def BBOX(self, pred, gt):
        h, w = pred.shape[-2:]

        y = torch.arange(0, h, dtype=torch.float)
        x = torch.arange(0, w, dtype=torch.float)
        y, x = torch.meshgrid(y, x)
        x = x.cuda()
        y=y.cuda()

        x_mask = (pred * x.unsqueeze(0))
        x_max = x_mask.flatten(1).max(-1)[0]
        x_min = x_mask.masked_fill(~(pred.type(torch.bool)), 1e8).flatten(1).min(-1)[0]

        y_mask = (pred * y.unsqueeze(0))
        y_max = y_mask.flatten(1).max(-1)[0]
        y_min = y_mask.masked_fill(~(pred.type(torch.bool)), 1e8).flatten(1).min(-1)[0]

        h1, w1 = gt.shape[-2:]

        y1 = torch.arange(0, h1, dtype=torch.float)
        x1 = torch.arange(0, w1, dtype=torch.float)
        y1, x1 = torch.meshgrid(y, x)

        x1_mask = (gt * x1.unsqueeze(0))
        x1_max = x1_mask.flatten(1).max(-1)[0]
        x1_min = x1_mask.masked_fill(~(gt.type(torch.bool)), 1e8).flatten(1).min(-1)[0]

        y1_mask = (gt * y1.unsqueeze(0))
        y1_max = y1_mask.flatten(1).max(-1)[0]
        y1_min = y1_mask.masked_fill(~(gt.type(torch.bool)), 1e8).flatten(1).min(-1)[0]

        loss = (abs(x_min-x1_min)+abs(y_min-y1_min))/2
        return loss

    '''
    def BBOX(self, img, gt):
        listx_pred = []
        listy_pred = []
        listx_gt = []
        listy_gt = []       
        
        pred = img.cpu().clone()
        pred = pred.squeeze(0)
        gt = gt.cpu().clone()
        gt = gt.squeeze(0)

        xy_pred = np.where(pred > 0.5)
        xy_gt = np.where(gt == 1)

        listx_pred = list(xy_pred)[2]
        listy_pred = list(xy_pred)[3]
        listx_gt = list(xy_gt)[2]
        listy_gt = list(xy_gt)[3]

        x_gt_min = min(listx_gt)
        y_gt_min = min(listy_gt)
        x_pred_min = min(listx_pred)
        y_pred_min = min(listy_pred)

        x_gt_max = max(listx_gt)
        y_gt_max = max(listy_gt)
        x_pred_max = max(listx_pred)
        y_pred_max = max(listy_pred)

        x_gt = (x_gt_max - x_gt_min)/2
        y_gt = (y_gt_max - y_gt_min)/2
        x_pred = (x_pred_max - x_pred_min)/2
        y_pred = (y_pred_max - y_pred_min)/2

        x = abs(x_pred - x_gt)
        y = abs(y_pred - y_gt)
        loss = x + y

        return loss


    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)

        img = pred.cpu().clone()
        img = img.squeeze(0)

        loss_iou = self.IOU(pred, mask)

        loss = loss_iou

        return loss

