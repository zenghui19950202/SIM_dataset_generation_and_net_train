#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2d fftshift and ifftshift functions designed for fft operation in old version of Pytorch
# author：zenghui time:2021/1/6
import torch

def torch_2d_ifftshift(image):
    image = image.squeeze()
    if image.dim() == 2:
        shift = [-(image.shape[1 - ax] // 2) for ax in range(image.dim())]
        image_shift = torch.roll(image, shift, [1, 0])
    else:
        raise Exception('The dim of image must be 2')
    return image_shift

def torch_2d_fftshift(image):
    image = image.squeeze()
    if image.dim() == 2:
        shift = [image.shape[ax] // 2 for ax in range(image.dim())]
        image_shift = torch.roll(image, shift, [0, 1])
    else:
        raise Exception('The dim of image must be 2')
    return image_shift