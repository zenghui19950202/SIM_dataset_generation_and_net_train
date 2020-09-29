#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/16

# sub-parts of the U-Net model

import torch
import torch.nn as nn
# from torchsummary import summary
import torch.nn.functional as F
from torchsummary import summary

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  if your machine do not have enough memory to handle all those weights
        #  bilinear interpolation could be used to do the upsampling.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,input_mode = 'only_input_SIM_images',LR_highway = None):
        super(UNet, self).__init__()
        self.input_mode = input_mode
        self.LR_highway = LR_highway
        if input_mode == 'only_input_SIM_images':
            self.inc = inconv(n_channels, 64)
        elif input_mode == 'input_all_images':
            self.inc = inconv(n_channels+1, 64)
        else:
            raise Exception("error Input mode")

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)

        self.outc5 = outconv(64, n_classes)

    def forward(self, x):
        if self.input_mode == 'only_input_SIM_images':
            x1 = self.inc(x[:,0:-1,:,:])
        elif self.input_mode == 'input_all_images':
            x1 = self.inc(x)
        else:
            raise Exception("error Input mode")
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        if self.LR_highway == 'add':
            y = self.outc5(y) + x[:,-1,:,:]
        elif self.LR_highway == 'concat':
            model_out = self.outc5(y)
            concat_data = torch.cat([model_out, x[:, -1, :, :].view(model_out.size())], 1)
            conv1x1 = torch.nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False)
            Tanh = torch.nn.Tanh()
            y = Tanh(conv1x1(concat_data))
        else:
            y = self.outc5(y)
        return y

if __name__ == '__main__':
    SIM_Unet = UNet(2,1,input_mode = 'input_all_images',LR_highway = 'concat')
    summary(SIM_Unet, input_size=(3, 256, 256))