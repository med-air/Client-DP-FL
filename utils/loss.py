import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, ret_mean=True, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.ret_mean = ret_mean
        self.activation = activation

    def dice_coef(self, pred, gt):
        """ computational formula
        """
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        elif self.activation == 'softmax':
            pred = torch.nn.functional.softmax(pred, dim=1)
        else:
            pass

        gt = gt.float()
        eps = 1e-5
        intersect = (pred * gt).sum((1, 2))
        y_sum = gt.sum((1, 2))
        z_sum = pred.sum((1, 2))
        dice = (2 * intersect + eps) / (z_sum + y_sum + eps)
        if self.ret_mean:
            return dice.mean()
        else:
            return dice

    def forward(self, pred, gt):
        if self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)
        elif self.activation == 'softmax':
            pred = torch.nn.functional.softmax(pred, dim=1)

        gt = gt.float()
        eps = 1e-5
        intersect = (pred * gt).sum((1, 2))
        y_sum = gt.sum((1, 2))
        z_sum = pred.sum((1, 2))
        loss = 1 - (2 * intersect + eps) / (z_sum + y_sum + eps)
        if self.ret_mean:
            return loss.mean()
        else:
            return loss


class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, gt):
       
        return self.ce(pred, gt) + self.dice(pred, gt)


def kd_loss(source_matrix, target_matrix):
     loss_fn = torch.nn.MSELoss(reduction='none')
     
     Q = source_matrix
     P = target_matrix
     loss = (F.kl_div(Q.log(), P, None, None, 'batchmean') + F.kl_div(P.log(), Q, None, None, 'batchmean'))/2.0
     return loss