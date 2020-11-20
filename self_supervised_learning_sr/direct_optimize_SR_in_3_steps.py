#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/30
'''The version of program that directly optimize the SR image in spatial domain:
I try to execute the optimization process at 3 direction respectively to avoid generating more unexpected
residual frequency peaks in Fourier frequency domain.
This method work well in simulation, but it seems that the optimization process on experimental data
is quite slow, and even cannot converge'''
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
from self_supervised_learning_sr import estimate_SIM_pattern
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
import torch.nn.functional as F

# import self_supervised_learning_sr.estimate_SIM_pattern
import numpy as np


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')
    return device


def train(net, SIM_data_loader, SIM_pattern_loader, net_input, criterion, num_epochs, device, lr=None,
          weight_decay=1e-5, opt_over='net',image_size = 256):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net = net.to(device)
    experimental_params = funcs.SinusoidalPattern(probability=1,image_size = image_size)


    # CTF = experimental_params.CTF_form(fc_ratio=2 + 2 * experimental_params.pattern_frequency_ratio)
    # CTF = CTF.to(device)

    # OTF_reconstruction = temp.OTF_form(fc_ratio=1 + temp.pattern_frequency_ratio)
    # psf_reconstruction = temp.psf_form(OTF_reconstruction)
    # psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction,device)

    noise = net_input.detach().clone()
    reg_noise_std = 0.03
    min_loss = 1e5
    image_size = [experimental_params.image_size, experimental_params.image_size]

    best_SR = torch.zeros(image_size, dtype=torch.float32, device=device)
    end_flag = 0

    input_id = 0
    data_id = 0
    for SIM_data, SIM_pattern in zip(SIM_data_dataloader, SIM_pattern_loader):
        SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1

    SR_LR = SIM_data[1]
    LR = SR_LR[:, :, :, 1].to(device)

    LR = torch.mean(SIM_raw_data[:,0:3,:,:],dim = 1).to(device)
    noise = torch.zeros_like(LR)
    input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data,[0,1,2,3,6])
    input_SIM_pattern = common_utils.pick_input_data(SIM_pattern,[0,1,2,3,6])

    if experimental_params.upsample == True:
        up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        SR_image0 = up_sample(copy.deepcopy(LR).unsqueeze(0)).to(device).squeeze()
        SR_image1 = up_sample(copy.deepcopy(LR).unsqueeze(0)).to(device).squeeze()
        SR_image2 = up_sample(copy.deepcopy(LR).unsqueeze(0)).to(device).squeeze()
        result_in_3_dim = torch.zeros([experimental_params.SR_image_size, experimental_params.SR_image_size, 3],
                                      device=device)
        OTF_upsmaple = experimental_params.OTF_upsmaple
        psf = experimental_params.psf_form(OTF_upsmaple)
        # SR_image_size = [experimental_params.SR_image_size, experimental_params.SR_image_size]
    else:
        SR_image0 = copy.deepcopy(LR).to(device)
        SR_image1 = copy.deepcopy(LR).to(device)
        SR_image2 = copy.deepcopy(LR).to(device)
        # SR_image0 = torch.rand_like(LR).to(device)
        # SR_image1 = torch.rand_like(LR).to(device)
        # SR_image2 = torch.rand_like(LR).to(device)

        result_in_3_dim = torch.zeros([experimental_params.image_size, experimental_params.image_size, 3],
                                      device=device)
        OTF = experimental_params.OTF
        psf = experimental_params.psf_form(OTF)

    psf_conv = funcs.psf_conv_generator(psf, device)

    OTF = experimental_params.OTF
    psf = experimental_params.psf_form(OTF)
    psf_radius = math.floor(psf.size()[0] / 2)
    mask = torch.zeros([1,1,image_size[0],image_size[1]],device = device)
    mask[:, :, psf_radius:-psf_radius, psf_radius:-psf_radius] = 1

    temp_input_SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels(
        input_SIM_raw_data)
    print(estimated_pattern_parameters)

    estimated_pattern_three_direction = torch.zeros([3,4],device =device)
    if estimated_pattern_parameters.size()[0] == 3:
        estimated_pattern_three_direction = estimated_pattern_parameters
    elif estimated_pattern_parameters.size()[0] == 5:
        estimated_pattern_three_direction[0, :] = torch.mean(estimated_pattern_parameters[0:3, :], dim=0)
        estimated_pattern_three_direction[1, :] = estimated_pattern_parameters[3, :]
        estimated_pattern_three_direction[2, :] = estimated_pattern_parameters[4, :]
    elif estimated_pattern_parameters.size()[0] == 9:
        estimated_pattern_three_direction[0, :] = torch.mean(estimated_pattern_parameters[0:3, :], dim=0)
        estimated_pattern_three_direction[1, :] = torch.mean(estimated_pattern_parameters[3:6, :], dim=0)
        estimated_pattern_three_direction[2, :] = torch.mean(estimated_pattern_parameters[6:9, :], dim=0)

    OTF_reconstruction = experimental_params.OTF_form(fc_ratio=1 + experimental_params.pattern_frequency_ratio)
    psf_reconstruction = experimental_params.psf_form(OTF_reconstruction)
    psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction, device)

    # experimental_parameters = SinusoidalPattern(probability=1)
    # xx, yy, _, _ = experimental_parameters.GridGenerate(image_size[0], grid_mode='pixel')

    SR_image0.requires_grad = True
    SR_image1.requires_grad = True
    SR_image2.requires_grad = True

    params0 = [{'params': SR_image0, 'weight_decay': weight_decay}]
    params1 = [{'params': SR_image1, 'weight_decay': weight_decay}]
    params2 = [{'params': SR_image2, 'weight_decay': weight_decay}]
    optimizer_pattern_params0 = optim.Adam(params0, lr=lr)
    optimizer_pattern_params1 = optim.Adam(params1, lr=lr)
    optimizer_pattern_params2 = optim.Adam(params2, lr=lr)

    SR_image = [SR_image0, SR_image1, SR_image2]

    optimizer = [optimizer_pattern_params0, optimizer_pattern_params1, optimizer_pattern_params2]

    input_SIM_raw_data = input_SIM_raw_data.to(device)
    input_SIM_pattern = input_SIM_pattern.to(device)
    temp_input_SIM_pattern = temp_input_SIM_pattern.to(device)
    reg_noise_std = 0.03

    for epoch in range(num_epochs):

        net_input_noise = noise.normal_() * reg_noise_std
        net_input_noise = net_input_noise.to(device)

        i = int(epoch*3 / num_epochs)
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer[i].zero_grad()
        SR_image_noise = SR_image[i] +  net_input_noise
        # SR_image_noise = SR_image[i]
        # LR_estimated = forward_model.positive_propagate((SR_image[i]+net_input_noise), 1, psf_conv,down_sample = experimental_params.upsample)
        # loss += criterion(LR_estimated, LR, mask)
        if temp_input_SIM_pattern.size()[1] == 3:
            SIM_raw_data_estimated = forward_model.positive_propagate(SR_image_noise,
                                                                      temp_input_SIM_pattern[:, i, :, :].detach(),
                                                                      psf_conv,down_sample = experimental_params.upsample)
            loss += criterion(SIM_raw_data_estimated, input_SIM_raw_data[:, i, :, :], mask)

            LR_estimated = forward_model.positive_propagate(SR_image[i],
                                                            1,
                                                            psf_conv,
                                                            down_sample=experimental_params.upsample)
            loss += criterion(LR_estimated, LR, mask)
        elif temp_input_SIM_pattern.size()[1] == 5:
            if i ==0:
                SIM_raw_data_estimated = forward_model.positive_propagate(SR_image_noise,
                                                                          temp_input_SIM_pattern[:, 0:3, :, :].detach(),
                                                                          psf_conv,down_sample = experimental_params.upsample)
                loss += criterion(SIM_raw_data_estimated, input_SIM_raw_data[:, 0:3, :, :], mask)
            else:
                SIM_raw_data_estimated = forward_model.positive_propagate(SR_image_noise,
                                                                          temp_input_SIM_pattern[:, i+ 2, :, :].detach(),
                                                                          psf_conv,down_sample = experimental_params.upsample)
                loss += criterion(SIM_raw_data_estimated, input_SIM_raw_data[:, 2 + i, :, :], mask)

                LR_estimated = forward_model.positive_propagate(SR_image[i],
                                                                          1,
                                                                          psf_conv,
                                                                          down_sample=experimental_params.upsample)
                loss += criterion(LR_estimated, LR, mask)
        elif temp_input_SIM_pattern.size()[1] == 9:
            SIM_raw_data_estimated = forward_model.positive_propagate(SR_image_noise,
                                                                      temp_input_SIM_pattern[:, i*3:3 + i*3, :, :],
                                                                      psf_conv,down_sample = experimental_params.upsample)
            loss += criterion(SIM_raw_data_estimated, input_SIM_raw_data[:, i*3:3 + i*3, :, :], mask)


        loss.backward()
        optimizer[i].step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, direction: %d train_loss: %f' % (epoch + 1, num_epochs,i, train_loss))

        # if min_loss > train_loss:
        #     min_loss = train_loss
        #     # SR_image_high_freq_and_notch_filtered = processing_utils.notch_filter(SR_image_high_freq_filtered, estimated_pattern_parameters)
        #     # best_SR = SR_image_high_freq_and_notch_filtered[psf_radius: -psf_radius, psf_radius: -psf_radius]
        #     best_SR = SR_image_high_freq_filtered.squeeze()[psf_radius: -psf_radius, psf_radius: -psf_radius]
        # if (epoch + 1) % 499 == 0:
        #     # result = SR_image_high_freq_filtered.squeeze()[psf_radius: -psf_radius, psf_radius: -psf_radius]
        #     SR_image_high_freq_and_notch_filtered = processing_utils.notch_filter_single_direction(
        #         SR_image_high_freq_filtered, estimated_pattern_three_direction[i, :])
        #     # common_utils.plot_single_tensor_image(SR_image_high_freq_and_notch_filtered)
        #     # common_utils.plot_single_tensor_image(SR_image_high_freq_filtered)
    for i in range(3):
        SR_image_high_freq_filtered = forward_model.positive_propagate(SR_image[i], 1, psf_reconstruction_conv)
        result_in_3_dim[:,:,i] = processing_utils.notch_filter_single_direction(
        SR_image_high_freq_filtered, estimated_pattern_three_direction[i, :])
    result = torch.mean(result_in_3_dim, dim=2)[psf_radius: -psf_radius, psf_radius: -psf_radius]
    common_utils.plot_single_tensor_image(result)
    return train_loss, result


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def train_net():
    pass


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
    # image_size = train_net_parameters['image_size']
    opt_over = train_net_parameters['opt_over']



    param_grid = {
        'learning_rate': [0.001],
        'batch_size': [1],
        'weight_decay': [1e-5],
        'Dropout_ratio': [1]
    }

    SIM_data = SpeckleSIMDataLoad.SIM_data_load(train_directory_file, normalize=False, data_mode='only_raw_SIM_data')
    SIM_pattern = SpeckleSIMDataLoad.SIM_pattern_load(train_directory_file, normalize=False)
    # SIM_pattern = SIM_data_load(train_directory_file, normalize=False, data_mode='only_raw_SIM_data')

    input_id = 1
    data_id = 0
    for a in SIM_data:
        SIM_raw_data = a[0]
        if data_id == input_id:
            break
        data_id += 1
    image_size = SIM_raw_data.size()[1]
    SIM_data_dataloader = DataLoader(SIM_data, batch_size=1)
    SIM_pattern_dataloader = DataLoader(SIM_pattern, batch_size=1)

    random.seed(60)  # 设置随机种子
    # min_loss = 1e5
    num_epochs = 9000

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    # criterion = nn.MSELoss()
    criterion = loss_functions.MSE_loss()
    num_raw_SIMdata, output_nc, num_downs = 2, 1, 5
    # SIMnet = Unet_for_self_supervised.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,input_mode = 'only_input_SIM_images', use_dropout=False)
    # SIMnet = Networks_Unet_GAN.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,
    #                                                 input_mode='only_input_SIM_images', use_dropout=False)
    SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, 1, input_mode='input_all_images', LR_highway=False)

    start_time = time.time()

    net_input = common_utils.get_noise(num_raw_SIMdata + 1, 'noise', (image_size, image_size), var=0.1)
    net_input = net_input.to(device).detach()

    train_loss, best_SR = train(SIMnet, SIM_data_dataloader, SIM_pattern_dataloader, net_input, criterion, num_epochs,
                                device, lr, weight_decay, opt_over,image_size)
    best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    common_utils.save_image_tensor2pillow(best_SR, save_file_directory)
    # SIMnet.to('cpu')
    end_time = time.time()
    # torch.save(SIMnet.state_dict(), file_directory + '/SIMnet.pkl')
    print(
        'avg train rmse: %f, learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
        % (train_loss, lr, batch_size, weight_decay, Dropout_ratio, end_time - start_time))
