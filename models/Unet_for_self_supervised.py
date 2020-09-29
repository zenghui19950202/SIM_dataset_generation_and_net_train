#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/27

import torch
import torch.nn as nn
import functools
from torchsummary import summary

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, LR_highway = None, use_dropout=False,input_mode = 'only_input_SIM_images'):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        self.double_conv = double_conv
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        if input_mode == 'only_input_SIM_images':
            self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                 norm_layer=norm_layer,LR_highway = LR_highway,input_mode = input_mode)  # add the outermost layer
        elif input_mode == 'input_all_images':
            self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc+1, submodule=unet_block,
                                                 outermost=True,
                                                 norm_layer=norm_layer,
                                                 LR_highway=LR_highway ,input_mode = input_mode)  # add the outermost layer
        else:
            raise Exception("error Input mode")
    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2, True)
        )
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, LR_highway = None, use_dropout=False,input_mode = 'only_input_SIM_images'):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.LR_highway = LR_highway
        self.outermost = outermost
        self.input_mode = input_mode
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        # downrelu = nn.LeakyReLU(0.2, True)
        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        downrelu = nn.LeakyReLU(0.2,True)
        uprelu = nn.LeakyReLU(0.2,True)
        downnorm = norm_layer(inner_nc)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, downrelu]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            # up = [uprelu, upconv, upnorm]
            up = [uprelu, upconv]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # down = [downrelu, downconv, downnorm]
            # up = [uprelu, upconv, upnorm]
            down = [downrelu, downconv]
            up = [uprelu, upconv]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(use_dropout)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            if self.input_mode == 'only_input_SIM_images':
                if self.LR_highway == 'add':
                    y = self.model(x[:,0:-1,:,:])
                    return y + x[:, -1, :, :].view(y.size())
                elif self.LR_highway == 'concat':
                    model_out = self.model(x[:,0:-1,:,:])
                    concat_data = torch.cat([model_out, x[:,-1,:,:].view(model_out.size())], 1)
                    conv1x1 = torch.nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False)
                    Tanh = torch.nn.Tanh()
                    out = Tanh(conv1x1(concat_data))
                    return out
                else:
                    return self.model(x[:,0:-1,:,:])
            elif self.input_mode == 'input_all_images':
                if self.LR_highway == 'add':
                    y = self.model(x)
                    return  y + x[:,-1,:,:].view(y.size())
                elif self.LR_highway == 'concat':
                    model_out = self.model(x)
                    concat_data = torch.cat([model_out, x[:,-1,:,:].view(model_out.size())], 1)
                    conv1x1 = torch.nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False)
                    Tanh = torch.nn.Tanh()
                    out = Tanh(conv1x1(concat_data))
                    return out
                else:
                    return self.model(x)
            else:
                raise Exception("error Input mode")
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)

if __name__ == '__main__':
    input_nc , output_nc ,num_downs = 9, 1, 6
    SIM_Unet = UnetGenerator(input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,input_mode = 'only_input_SIM_images',LR_highway = 'False',use_dropout=False)
    summary(SIM_Unet, input_size=(10, 256, 256),device="cpu")