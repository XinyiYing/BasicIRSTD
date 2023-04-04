# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch


def adjust_learning_rate(optimizer, epoch, epochs, lr, warm_up_epochs=0, min_lr=0):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""

    if epoch < warm_up_epochs:
        cur_lr = lr * epoch / warm_up_epochs
    else:
        cur_lr = pow(1 - float(epoch - warm_up_epochs) / (epochs - warm_up_epochs + 1), 0.9) \
                 * (lr - min_lr) + min_lr



    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr