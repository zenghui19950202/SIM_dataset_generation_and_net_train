#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/21

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
          weight_decay=1e-5, opt_over='net'):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net = net.to(device)
    temp = funcs.SinusoidalPattern(probability=1)
    OTF = temp.OTF
    psf = temp.psf_form(OTF)

    psf_conv = funcs.psf_conv_generator(psf,device)

    CTF = temp.CTF_form(fc_ratio=2 + 2*temp.pattern_frequency_ratio)
    CTF = CTF.to(device)


    # OTF_reconstruction = temp.OTF_form(fc_ratio=1 + temp.pattern_frequency_ratio)
    # psf_reconstruction = temp.psf_form(OTF_reconstruction)
    # psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction,device)


    noise = net_input.detach().clone()
    reg_noise_std = 0.03
    min_loss = 1e5
    image_size = [net_input.size()[2], net_input.size()[3]]
    best_SR = torch.zeros(image_size, dtype=torch.float32, device=device)
    end_flag = 0

    input_id = 0
    data_id = 0
    for SIM_data, SIM_pattern in zip(SIM_data_dataloader, SIM_pattern_loader):
        SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1


    input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data,[3,4,5])
    input_SIM_pattern = common_utils.pick_input_data(SIM_pattern,[3,4,5])
    # input_SIM_raw_data_normalized = processing_utils.pre_processing(input_SIM_raw_data)

    # SR_image = SIM_data[1][:,:,:,1]
    wide_field_image = torch.mean(input_SIM_raw_data[:,0:3,:,:],dim=1)
    wide_field_image = wide_field_image / wide_field_image.max()
    # SR_image = forward_model.winier_deconvolution(wide_field_image,OTF)
    # SR_image = torch.rand_like(wide_field_image)
    SR_image = wide_field_image
    psf_radius = math.floor(psf.size()[0] / 2)

    mask = torch.zeros_like(input_SIM_raw_data, device=device)
    mask[:, :, psf_radius:-psf_radius, psf_radius:-psf_radius] = 1

    temp_input_SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels(
        input_SIM_raw_data)
    print(estimated_pattern_parameters)

    OTF_reconstruction = temp.OTF_form(fc_ratio=1 + temp.pattern_frequency_ratio)
    psf_reconstruction = temp.psf_form(OTF_reconstruction)
    psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction,device)


    experimental_parameters = SinusoidalPattern(probability=1)
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size[0], grid_mode='pixel')

    delta_pattern_params = torch.zeros_like(estimated_pattern_parameters)

    delta_pattern_params.requires_grad = True


    params = []
    SR_image = SR_image.to(device)
    SR_image.requires_grad = True
    params += [{'params': SR_image, 'weight_decay': weight_decay}]
    params += [{'params': delta_pattern_params, 'weight_decay': weight_decay}]
    optimizer_pattern_params = optim.Adam(params, lr=0.001)

    input_SIM_raw_data = input_SIM_raw_data.to(device)
    input_SIM_pattern = input_SIM_pattern.to(device)
    temp_input_SIM_pattern = temp_input_SIM_pattern.to(device)
    # temp_input_SIM_pattern = input_SIM_pattern
    # temp_input_SIM_pattern = input_SIM_pattern
    # delta_pattern_params = estimated_pattern_parameters
    reg_noise_std = 0.03
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode

        net_input_noise = wide_field_image.normal_() * reg_noise_std
        net_input_noise = net_input_noise.to(device)

        optimizer_pattern_params.zero_grad()
        # temp_input_SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern(temp_input_SIM_pattern,
        #                                              estimated_pattern_parameters,
        #                                              delta_pattern_params, xx, yy)
        # SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern(SIM_raw_data.shape, estimated_pattern_parameters,
        #                                                            delta_pattern_params, xx, yy)

        SIM_raw_data_estimated = forward_model.positive_propagate(abs(SR_image+ net_input_noise), temp_input_SIM_pattern, psf_conv)
        SR_image_high_freq_filtered = forward_model.positive_propagate(SR_image.detach(), 1, psf_reconstruction_conv)
        mse_loss = criterion(SIM_raw_data_estimated,input_SIM_raw_data, mask)

        # SR_complex = torch.stack([SR_image, torch.zeros_like(SR_image)], 2)
        # FFT_SR = torch.fft(SR_complex, 2 )
        # shifted_FFT_SR = forward_model.torch_2d_fftshift(FFT_SR)
        # shifted_FFT_SR_intensity = pow( pow(shifted_FFT_SR[:,:,0],2) + pow(shifted_FFT_SR[:,:,1],2), 1/2)
        # normalized_shifted_FFT_SR = torch.log(1+(shifted_FFT_SR_intensity))
        # tv_loss = 1e-8 * loss_functions.tv_loss_calculate(normalized_shifted_FFT_SR)
        #
        # # tv_loss = 1e-6 * loss_functions.tv_loss_calculate(abs(SR_image))
        #
        # loss = mse_loss + tv_loss
        loss = mse_loss
        loss.backward()
        optimizer_pattern_params.step()


        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))
        # SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern(SIM_raw_data.shape,estimated_pattern_parameters,delta_pattern_params,xx,yy)
        # print(delta_pattern_params)

        if min_loss > train_loss:
            min_loss = train_loss
            # SR_image_high_freq_and_notch_filtered = processing_utils.notch_filter(SR_image_high_freq_filtered, estimated_pattern_parameters)
            # best_SR = SR_image_high_freq_and_notch_filtered[psf_radius: -psf_radius, psf_radius: -psf_radius]
            best_SR =SR_image_high_freq_filtered.squeeze()[psf_radius: -psf_radius, psf_radius: -psf_radius]
        if (epoch + 1) % 500 == 0:

            # SR_image_high_freq_and_notch_filtered = processing_utils.notch_filter(SR_image_high_freq_filtered, estimated_pattern_parameters)
            # result = SR_image_high_freq_and_notch_filtered[psf_radius: -psf_radius, psf_radius: -psf_radius]
            #
            # SR_image_notch_filtered = processing_utils.notch_filter(SR_image, estimated_pattern_parameters)
            # result1 = forward_model.low_pass_filter(SR_image_notch_filtered.to(device),CTF)
            # result1 = result1[psf_radius: -psf_radius, psf_radius: -psf_radius]

            result = SR_image_high_freq_filtered.squeeze()[psf_radius: -psf_radius, psf_radius: -psf_radius]

            common_utils.plot_single_tensor_image(result)
            # common_utils.plot_single_tensor_image(SR_image_high_freq_filtered)

    return train_loss, best_SR

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
    image_size = train_net_parameters['image_size']
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

    SIM_data_dataloader = DataLoader(SIM_data, batch_size=1)
    SIM_pattern_dataloader = DataLoader(SIM_pattern, batch_size=1)

    random.seed(60)  # 设置随机种子
    # min_loss = 1e5
    num_epochs = 1200

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

    net_input = common_utils.get_noise(num_raw_SIMdata+1, 'noise', (image_size, image_size), var=0.1)
    net_input = net_input.to(device).detach()


    train_loss, best_SR = train(SIMnet, SIM_data_dataloader, SIM_pattern_dataloader, net_input, criterion, num_epochs,
                                device, lr, weight_decay, opt_over)
    best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    common_utils.save_image_tensor2pillow(best_SR, save_file_directory)
    # SIMnet.to('cpu')
    end_time = time.time()
    # torch.save(SIMnet.state_dict(), file_directory + '/SIMnet.pkl')
    print(
        'avg train rmse: %f, learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
        % (train_loss, lr, batch_size, weight_decay, Dropout_ratio, end_time - start_time))

