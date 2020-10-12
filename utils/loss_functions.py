#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/10
import torch
import torch.nn as nn
# from utils.common_utils import plot_single_tensor_image
class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pattern, pattern_gt, mask):

        num_of_channels = pattern.size()[1]
        if pattern.mean() < 1e-20:
            pattern_nomalized = pattern / (pattern.mean() + 1e-19)
        else:
            pattern_nomalized = pattern / pattern.mean()
        if pattern_gt.mean() < 1e-20:
            pattern_gt_nomalized = pattern_gt / (pattern_gt.mean() + 1e-19)
        else:
            pattern_gt_nomalized = pattern_gt / pattern_gt.mean()
        loss = torch.mean(torch.pow((pattern_nomalized - pattern_gt_nomalized)* mask, 2))
        return loss * num_of_channels

def tv_loss_calculate(x, beta=0.5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))


def minus_loss_calculate(x):
    minus_x = torch.mean(abs(x) - x)

    return minus_x