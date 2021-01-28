#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Directly reconstruct the super resolution image and the polarization distribution using gradient descent
# author：zenghui time:2021/1/27


from parameter_estimation import *
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

import numpy as np
import matplotlib.pyplot as plt


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')
    return device

def compare_reconstruction_quality_using_different_input_frames(SIM_data_dataloader,image_show = False):
    data_num = 0
    for SIM_data in SIM_data_dataloader:
        SIM_raw_data = SIM_data[0]
        SSIM_of_diff_input_num = torch.zeros(5,1)
        PSNR_of_diff_input_num = torch.zeros(5,1)
        for i in range(5):
            SSIM_of_diff_input_num[i,0],PSNR_of_diff_input_num[i,0],SR = SR_reconstruction(SIM_data,input_num = i+5,image_show = image_show)
        if data_num == 0:
            SSIM = SSIM_of_diff_input_num
            PSNR = PSNR_of_diff_input_num
        else:
            SSIM = torch.cat([SSIM, SSIM_of_diff_input_num], 1)
            PSNR = torch.cat([PSNR, PSNR_of_diff_input_num], 1)
        data_num += 1
        if data_num > 30:
            break

    SSIM_mean = torch.mean(SSIM,1).numpy()
    SSIM_std = torch.std(SSIM, 1).numpy()

    PSNR_mean = torch.mean(PSNR,1).numpy()
    PSNR_std = torch.std(PSNR, 1).numpy()

    np.save(save_file_directory +"SSIM.npy", SSIM.numpy())
    np.save(save_file_directory +"PSNR.npy", PSNR.numpy())

    index = np.arange(5)
    total_width, n = 0.4, 2
    width = total_width / n

    plt.title('A Bar Chart')
    plt.bar(index, SSIM_mean, width=width, yerr=SSIM_std, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='SSIM',color='#583d72')
    plt.legend(loc=2)
    plt.savefig(save_file_directory + 'SSIM_bar.eps', dpi=600, format='eps')
    plt.show()
    plt.bar(index-width, PSNR_mean, width=width, yerr=PSNR_std, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7,
            label='PSNR',color = '#9f5f80')
    plt.xticks(index + 0.2, ['5', '6', '7', '8', '9'])
    plt.legend(loc=2)
    plt.grid(linestyle='--',c='#bbbbbb')
    plt.savefig(save_file_directory + 'PSNR_bar.eps', dpi=600,format='eps')
    plt.show()

def SR_reconstruction( SIM_data,input_num = 5,image_show = True):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)

    LR_HR = SIM_data[1]
    SIM_raw_data = SIM_data[0]

    image_size = [SIM_raw_data.size()[2],SIM_raw_data.size()[3]]
    experimental_params = funcs.SinusoidalPattern(probability=1,image_size = image_size[0])

    HR = LR_HR[:,:,:,0]

    OTF = experimental_params.OTF
    psf = experimental_params.psf_form(OTF)
    OTF = OTF.to(device)
    CTF = experimental_params.CTF_form(fc_ratio=1).to(device)

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
    elif input_num == 4:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 3, 6,1])
    elif input_num == 1:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [2])
    # input_SIM_pattern = common_utils.pick_input_data(SIM_pattern)

    # input_SIM_raw_data_normalized = processing_utils.pre_processing(input_SIM_raw_data)

    wide_field_image = torch.mean(input_SIM_raw_data[:, 0:3, :, :], dim=1)
    wide_field_image = wide_field_image / wide_field_image.max()
    # SR_image = forward_model.winier_deconvolution(wide_field_image,OTF)
    # SR_image = torch.rand_like(wide_field_image)

    if experimental_params.upsample == True:
        up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        SR_image = up_sample(copy.deepcopy(wide_field_image).unsqueeze(0))
        HR = up_sample(HR.unsqueeze(0))
    else:
        SR_image = copy.deepcopy(wide_field_image)
    HR = HR / HR.max()
    HR = HR.squeeze().numpy() * 255

    temp_input_SIM_pattern, estimated_pattern_parameters,estimated_SIM_pattern_without_m = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
        input_SIM_raw_data)

    print(estimated_pattern_parameters)
    temp_input_SIM_pattern = temp_input_SIM_pattern.to(device)
    estimated_pattern_parameters = estimated_pattern_parameters.to(device)
    estimated_modulation_factor = estimated_pattern_parameters[:,2].clone().detach().to(device)
    estimated_modulation_factor.requires_grad = True

    params = []
    SR_image = SR_image.to(device)
    SR_image.requires_grad = True
    polarization_direction = torch.rand(SR_image.size()).to(device)
    polarization_direction.requires_grad = True

    params += [{'params': SR_image, 'weight_decay': weight_decay}]
    params += [{'params': polarization_direction, 'weight_decay': weight_decay}]
    params += [{'params': estimated_modulation_factor}]
    optimizer_SR_and_polarization = optim.Adam(params, lr=0.001)

    input_SIM_raw_data = input_SIM_raw_data.to(device)

    input_SIM_raw_data_fft = unsample_process(image_size,input_SIM_raw_data,CTF,experimental_params.upsample)

    SR_reuslt = reconstruction(SR_image, polarization_direction, input_SIM_raw_data_fft, optimizer_SR_and_polarization,
                                        temp_input_SIM_pattern,estimated_pattern_parameters, experimental_params,image_show=image_show)
    SR = SR_reuslt / SR_reuslt.max()
    SR_np = SR.cpu().squeeze().detach().numpy()*255
    SSIM = SRimage_metrics.calculate_ssim(SR_np,HR)
    PSNR = SRimage_metrics.calculate_psnr_np(SR_np,HR)
    # SSIM = 1
    # PSNR = 1

    return SSIM,PSNR,SR #(1+torch.cos(polarization_direction))


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

def unsample_process(image_size,input_SIM_raw_data,CTF,upsample_flag=False):
    if upsample_flag == True:
        if image_size[0] % 2 ==0:
            padding_size = int(image_size[0] / 2)
            ZeroPad = nn.ZeroPad2d(padding=(padding_size, padding_size, padding_size, padding_size))

        else:
            padding_size = int(image_size[0] / 2)
            ZeroPad = nn.ZeroPad2d(padding=(padding_size+1, padding_size, padding_size+1, padding_size))
        input_SIM_raw_data_fft = torch.zeros([1, 9, image_size[0] * 2, image_size[1] * 2, 2], device=device)
        for i in range(input_SIM_raw_data.size()[1]):
            input_SIM_raw_data_complex = torch.stack(
                [input_SIM_raw_data[:, i, :, :].squeeze(), torch.zeros_like(input_SIM_raw_data[:, i, :, :]).squeeze()],
                2)
            SIM_raw_data_fft = forward_model.torch_2d_fftshift(
                torch.fft(input_SIM_raw_data_complex, 2)) * CTF.unsqueeze(2)

            input_SIM_raw_data_fft[:, i, :, :, :] = torch.stack([ZeroPad(SIM_raw_data_fft[:,:,0]),ZeroPad(SIM_raw_data_fft[:,:,1])],2)
    else:
        input_SIM_raw_data_fft = torch.zeros([1, 9,image_size[0], image_size[1],2], device=device)
        for i in range(input_SIM_raw_data.size()[1]):
            input_SIM_raw_data_complex = torch.stack(
                [input_SIM_raw_data[:, i, :, :].squeeze(), torch.zeros_like(input_SIM_raw_data[:, i, :, :]).squeeze()], 2)
            input_SIM_raw_data_fft[:,i,:,:,:] = forward_model.torch_2d_fftshift(
                torch.fft(input_SIM_raw_data_complex, 2))* CTF.unsqueeze(2)

    return input_SIM_raw_data_fft

def reconstruction(SR_image,polarization_direction, input_SIM_raw_data_fft,optimizer_SR_and_polarization, input_SIM_pattern,estimated_pattern_parameters,experimental_params,image_show=True):
    device = SR_image.device
    OTF = experimental_params.OTF_upsmaple.to(device)
    for epoch in range(num_epochs):
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer_SR_and_polarization.zero_grad()
        for i in range(estimated_pattern_parameters.size()[0]):
            theta = torch.atan(estimated_pattern_parameters[i, 1] / estimated_pattern_parameters[i, 0])
            # sample_light_field = SR_image * input_SIM_pattern[:, i, :, :] * (1+0.6* torch.cos(theta-polarization_direction))
            sample_light_field = SR_image * input_SIM_pattern[:, i, :, :]
            sample_light_field_complex = torch.stack([sample_light_field.squeeze(), torch.zeros_like(sample_light_field).squeeze()], 2)
            SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                torch.fft((sample_light_field_complex), 2)) * OTF.unsqueeze(2)
            mse_loss = criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:,i,:,:,:], 1, OTF, normalize=True, deconv=False)
            loss += mse_loss
        loss.backward()
        optimizer_SR_and_polarization.step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    if image_show == True:
        result = processing_utils.notch_filter_for_all_vulnerable_point(SR_image,estimated_pattern_parameters,experimental_params).squeeze()
        common_utils.plot_single_tensor_image(result)

    return result

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
    num_epochs = 800

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    # criterion = nn.MSELoss()
    criterion = loss_functions.MSE_loss_2()
    num_raw_SIMdata, output_nc, num_downs = 2, 1, 5

    # compare_reconstruction_quality_using_different_input_frames(SIM_data_dataloader,image_show = False)

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

