import torch
import torch.nn as nn

# class Loss:
#     pass


# Loss = torch.nn.SmoothL1Loss()

# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()
#         self.loss = torch.nn.MSELoss()
#
#     def forward(self, x):
#         return self.loss(x)

# Loss = torch.nn.MSELoss

class FL(nn.Module):
    def forward(self, preds, targets):
        return _neg_loss(preds, targets)

def _neg_loss(preds, targets):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      preds (B x c x h x w)
      gt_regr (B x c x h x w)
  '''
  pos_inds = targets.eq(1).float()
  neg_inds = targets.lt(1).float()

  neg_weights = torch.pow(1 - targets, 4)

  loss = 0
  for pred in preds:
    # pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)

class Focalloss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, aver = False):
        super(Focalloss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.aver = aver

    def forward(self, pred, target, weights = None):
        pos_inds = target.eq(1).float()
        neg_inds = target.eq(0).float()
        pos_loss = -self.alpha*torch.log(pred) * torch.pow(1 - pred, self.gamma) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        loss = pos_loss + neg_loss

        if self.aver:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        # Old One
        pred = pred.squeeze(dim=1)
        # pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -
                                                intersection.sum() + smooth)
        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


# LossPoint = FL
# Loss = LossPoint
# Loss2 = Focalloss
Loss = torch.nn.MSELoss
Loss2 = SoftIoULoss

