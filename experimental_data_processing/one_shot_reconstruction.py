#!/usr/bin/env python
# -*- coding: utf-8 -*-
# objection: To reconstruct the SIM SR image using original least squares
# problem: There is no fuction that can be used to construct the toeplitz matrix directly
# author：zenghui time:2020/12/24

from utils import *
from models import *
from self_supervised_learning_sr import *
import torch
import torch.optim as optim
import torch.nn as nn
import random
import time
import copy
import math
from torch.utils.data import DataLoader
from simulation_data_generation import fuctions_for_generate_pattern as funcs
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from simulation_data_generation import SRimage_metrics
from parameter_estimation import estimate_SIM_pattern

# import self_supervised_learning_sr.estimate_SIM_pattern
import numpy as np
import matplotlib.pyplot as plt


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')
    return device


def SR_reconstruction( SIM_data,input_num = 5,image_show = True):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)

    LR_HR = SIM_data[1]
    SIM_raw_data = SIM_data[0]
    HR = LR_HR[:,:,:,0]
    HR = HR / HR.max()
    HR = HR.squeeze().numpy()*255

    image_size = [SIM_raw_data.size()[2],SIM_raw_data.size()[3]]
    experimental_params = funcs.SinusoidalPattern(probability=1,image_size = image_size[0])
    OTF = experimental_params.OTF
    psf = experimental_params.psf_form(OTF)
    OTF = OTF.to(device)
    CTF = experimental_params.CTF_form(fc_ratio=2).to(device)

    if input_num == 5:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data,[0, 1 ,2 ,3 ,6])
    elif input_num == 6:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6, 4])
    elif input_num == 7:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6 ,4 ,7])
    elif input_num == 8:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6 ,4 ,7 ,5])
    elif input_num == 9:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data)
    # input_SIM_pattern = common_utils.pick_input_data(SIM_pattern)

    # input_SIM_raw_data_normalized = processing_utils.pre_processing(input_SIM_raw_data)

    wide_field_image = torch.mean(input_SIM_raw_data[:, 0:3, :, :], dim=1)
    wide_field_image = wide_field_image / wide_field_image.max()
    # SR_image = forward_model.winier_deconvolution(wide_field_image,OTF)
    # SR_image = torch.rand_like(wide_field_image)

    if experimental_params.upsample == True:
        up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        SR_image_step1 = up_sample(copy.deepcopy(wide_field_image).unsqueeze(0))
        SR_image_step2 = up_sample(copy.deepcopy(wide_field_image).unsqueeze(0))
        OTF = experimental_params.OTF_upsmaple.to(device)
    else:
        SR_image_step1 = copy.deepcopy(wide_field_image)
        SR_image_step2 = copy.deepcopy(wide_field_image)

    temp_input_SIM_pattern, estimated_pattern_parameters,estimated_SIM_pattern_without_m = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
        input_SIM_raw_data)

    print(estimated_pattern_parameters)

    estimated_pattern_parameters = estimated_pattern_parameters.to(device)
    estimated_modulation_factor = estimated_pattern_parameters[:,2].clone().detach().to(device)
    estimated_modulation_factor.requires_grad = True

    OTF_reconstruction = experimental_params.OTF_form(fc_ratio=1 + experimental_params.pattern_frequency_ratio)
    psf_reconstruction = experimental_params.psf_form(OTF_reconstruction)
    psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction, device)





    input_SIM_raw_data = input_SIM_raw_data.to(device)

if __name__ == '__main__':
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    train_directory_file = train_net_parameters['train_directory_file']
    valid_directory_file = train_net_parameters['valid_directory_file']
    save_file_directory = train_net_parameters['save_file_directory']
    data_generate_mode = train_net_parameters['data_generate_mode']
    net_type = train_net_parameters['net_type']
    data_input_mode = train_net_parameters['data_input_mode']
    LR_highway_type = train_net_parameters['LR_highway_type']
    MAX_EVALS = train_net_parameters['MAX_EVALS']
    num_epochs = train_net_parameters['num_epochs']
    data_num = train_net_parameters['data_num']
    image_size = train_net_parameters['image_size']
    opt_over = train_net_parameters['opt_over']


    SIM_data = SpeckleSIMDataLoad.SIM_data_load(train_directory_file, normalize=False, data_mode='only_raw_SIM_data')
    SIM_pattern = SpeckleSIMDataLoad.SIM_pattern_load(train_directory_file, normalize=False)
    # SIM_pattern = SIM_data_load(train_directory_file, normalize=False, data_mode='only_raw_SIM_data')

    SIM_data_dataloader = DataLoader(SIM_data, batch_size=1)
    SIM_pattern_dataloader = DataLoader(SIM_pattern, batch_size=1)

    device = try_gpu()
    criterion = loss_functions.MSE_loss()

    start_time = time.time()

    input_id = 1
    data_id = 0
    for SIM_data, SIM_pattern in zip(SIM_data_dataloader, SIM_pattern_dataloader):
        # SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1

    SSIM, PSNR, best_SR = SR_reconstruction(SIM_data,input_num = 5)


    if not best_SR.dim() == 4:
        best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    common_utils.save_image_tensor2pillow(best_SR, save_file_directory)
    end_time = time.time()

    print(' SSIM:%f, PSNR:%f,time: %f ' % (SSIM, PSNR, end_time - start_time))