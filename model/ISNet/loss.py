import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftIoULoss(nn.Module):
    def __init__(self, batch=32):
        super(SoftIoULoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        loss1 = self.bce_loss(pred, target)
        return loss + loss1
class SoftIoULoss1(nn.Module):
    def __init__(self, batch=32):
        super(SoftIoULoss1, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        loss1 = self.bce_loss(pred, target)
        return loss
