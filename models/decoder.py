#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/27
import torch.nn as nn

class decoder_generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64):
        slope = 0.03
        decoder_net = nn.Sequential(
              nn.ConvTranspose2d(input_nc , outer_nc,
                                   kernel_size=4, stride=2,
                                   padding=1),
              nn.LeakyReLU(slope, True),

          nn.ReLU()
        )

