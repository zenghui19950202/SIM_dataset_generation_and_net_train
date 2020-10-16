#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/12

def positive_propagate(SR_image, SIM_pattern, psf_conv):
    SIM_raw_data_estimated = psf_conv(SR_image * SIM_pattern)
    return SIM_raw_data_estimated

def deconvolution(SR_image,OTF):
    pass