#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/12
from numpy.fft import fft2
from numpy.fft import fftshift
from numpy.fft import ifft2
from numpy.fft import ifftshift
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
import torch
import heapq

def pre_processing1(raw_SIM_data):
    _, channels, image_sizex, image_sizey = raw_SIM_data.shape
    raw_SIM_data_preprocess = torch.zeros_like(raw_SIM_data)
    for i in range(channels):
        raw_SIM_slice = raw_SIM_data[:, i, :, :].squeeze()
        raw_SIM_slice_min = raw_SIM_slice.min()
        raw_SIM_slice_no_bg = raw_SIM_slice - raw_SIM_slice_min

        raw_SIM_slice_no_bg_max = raw_SIM_slice_no_bg.max()
        if raw_SIM_slice_no_bg_max < 1e-20:
            raw_SIM_slice_no_bg_nomalized = raw_SIM_slice_no_bg / (raw_SIM_slice_no_bg_max + 1e-19)
        else:
            raw_SIM_slice_no_bg_nomalized = raw_SIM_slice_no_bg / raw_SIM_slice_no_bg_max

        raw_SIM_data_preprocess[:,i,:,:] = raw_SIM_slice_no_bg_nomalized
    return raw_SIM_data_preprocess

def pre_processing(raw_SIM_data):
    _, channels, image_sizex, image_sizey = raw_SIM_data.shape
    raw_SIM_data_preprocess = torch.zeros_like(raw_SIM_data)
    for i in range(channels):
        raw_SIM_slice = raw_SIM_data[:, i, :, :].squeeze()


        raw_SIM_slice_min_list = heapq.nsmallest(10, raw_SIM_slice.detach().view(1, -1).squeeze())
        raw_SIM_slice_min = torch.tensor([0.], device=raw_SIM_slice.device)

        for temp_min in raw_SIM_slice_min_list:
            raw_SIM_slice_min += temp_min

        raw_SIM_slice_min = raw_SIM_slice_min / 10
        raw_SIM_slice_no_bg = raw_SIM_slice - raw_SIM_slice_min


        raw_SIM_slice_no_bg_max_list = heapq.nlargest(10, raw_SIM_slice_no_bg.detach().view(1, -1).squeeze())
        raw_SIM_slice_no_bg_max = torch.tensor([0.], device=raw_SIM_slice_no_bg.device)

        for temp_max in raw_SIM_slice_no_bg_max_list:
            raw_SIM_slice_no_bg_max += temp_max
        raw_SIM_slice_no_bg_max = raw_SIM_slice_no_bg_max / 10

        if raw_SIM_slice_no_bg_max < 1e-20:
            raw_SIM_slice_no_bg_nomalized = raw_SIM_slice_no_bg / (raw_SIM_slice_no_bg_max + 1e-19)
        else:
            raw_SIM_slice_no_bg_nomalized = raw_SIM_slice_no_bg / raw_SIM_slice_no_bg_max

        raw_SIM_data_preprocess[:,i,:,:] = raw_SIM_slice_no_bg_nomalized
    return raw_SIM_data_preprocess

def notch_filter(SR_image, estimated_pattern_parameters):
    SR_image = SR_image.squeeze()
    image_size = SR_image.size()
    SR_image_np = SR_image.detach().cpu().numpy()
    fft_image_np = fftshift(fft2(SR_image_np, axes=(0, 1)), axes=(0, 1))
    experimental_parameters = SinusoidalPattern(probability=1)
    fx, fy, _, _ = experimental_parameters.GridGenerate(image_size[0], grid_mode='pixel')
    spatial_freq = estimated_pattern_parameters[:, 0:2].squeeze()
    notch_filter = torch.zeros_like(SR_image)
    device = SR_image.device
    freq_list = [-1,1,-2,2]
    input_num = spatial_freq.size()[0]
    for i in range(3):
        if input_num == 9:
            spatial_freq_mean = torch.mean(spatial_freq[0 + 3 * i:3 + 3 * i, :], dim=0)
        elif input_num == 5:
            if i == 0:
                spatial_freq_mean = torch.mean(spatial_freq[0:3, :], dim=0)
            else:
                spatial_freq_mean = spatial_freq[2 + i, :]
        elif input_num == 4:
            spatial_freq_mean = spatial_freq[i, :]
        for j in freq_list:
            fx_shift = fx - j * spatial_freq_mean[0]
            fy_shift = fy - j * spatial_freq_mean[1]
            fr_square = (fx_shift ** 2 + fy_shift ** 2)
            f0 = image_size[0]/256 * 2
            notch_filter += torch.exp(-1 * fr_square / (4 * f0 *f0)).to(device)

    fft_image_np_filtered = fft_image_np * (1- notch_filter.detach().cpu().numpy())
    image_np_filtered = abs(ifft2(ifftshift(fft_image_np_filtered, axes=(0, 1)), axes=(0, 1)))
    image_np_filtered = image_np_filtered/image_np_filtered.max() * 256
    image_filtered_tensor = torch.from_numpy(image_np_filtered)

    return image_filtered_tensor