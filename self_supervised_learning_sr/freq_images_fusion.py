#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/12
from self_supervised_learning_sr.forward_model import positive_propagate
import torch

def image_fusion(high_freq_image, low_freq_image, beta, psf_conv, psf_reconstruction_conv):
    low_freq_image = low_freq_image.unsqueeze(0)
    high_freq_image = positive_propagate(high_freq_image,torch.ones_like(high_freq_image),psf_reconstruction_conv) - positive_propagate(high_freq_image,torch.ones_like(high_freq_image),psf_conv)

    SR_image = beta * high_freq_image + low_freq_image

    return SR_image
