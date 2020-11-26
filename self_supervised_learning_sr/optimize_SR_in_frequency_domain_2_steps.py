#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/11/12

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
from simulation_data_generation import SRimage_metrics

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

def compare_reconstruction_quality_using_different_input_frames(SIM_data_dataloader):
    for SIM_data in SIM_data_dataloader:
        SIM_raw_data = SIM_data[0]
        data_num = 0
        SSIM_of_diff_input_num = torch.zeros(5,1)
        PSNR_of_diff_input_num = torch.zeros(5,1)
        for i in range(5):
            SSIM_of_diff_input_num[i,0],PSNR_of_diff_input_num[i,0],SR = SR_reconstruction(SIM_raw_data,input_num = i+5)
        if i == 0:
            SSIM = SSIM_of_diff_input_num
            PSNR = PSNR_of_diff_input_num
        else:
            SSIM = torch.cat([SSIM, SSIM_of_diff_input_num], 1)
            PSNR = torch.cat([PSNR, PSNR_of_diff_input_num], 1)
        data_num += 1
        if data_num > 2:
            break

    SSIM_mean = torch.mean(SSIM,1).numpy()
    SSIM_std = torch.std(SSIM, 1).numpy()

    PSNR_mean = torch.mean(PSNR,1).numpy()
    PSNR_std = torch.std(PSNR, 1).numpy()

    index = np.arange(5)
    plt.title('A Bar Chart')
    plt.bar(index, SSIM_mean, yerr=SSIM_std, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='First')
    plt.xticks(index + 0.2, ['5', '6', '7', '8', '9'])
    plt.legend(loc=2)
    plt.show()

def SR_reconstruction( SIM_data,input_num = 5):
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
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data,[0,1,2,3,6])
    elif input_num == 6:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6, 4])
    elif input_num == 7:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6 ,4 ,7])
    elif input_num == 8:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6 ,4 ,7 ,5])
    elif input_num == 9:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data)
    input_SIM_pattern = common_utils.pick_input_data(SIM_pattern)

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
    estimated_SIM_pattern_without_m = estimated_SIM_pattern_without_m.to(device)
    print(estimated_pattern_parameters)
    estimated_modulation_factor = estimated_pattern_parameters[:,2].clone().detach().to(device)
    estimated_modulation_factor.requires_grad = True

    OTF_reconstruction = experimental_params.OTF_form(fc_ratio=1 + experimental_params.pattern_frequency_ratio)
    psf_reconstruction = experimental_params.psf_form(OTF_reconstruction)
    psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction, device)

    params = []
    SR_image_step1 = SR_image_step1.to(device)
    SR_image_step1.requires_grad = True
    SR_image_step2 = SR_image_step2.to(device)
    SR_image_step2.requires_grad = True
    params += [{'params': SR_image_step1, 'weight_decay': weight_decay}]
    params += [{'params': SR_image_step2, 'weight_decay': weight_decay}]
    params += [{'params': estimated_modulation_factor}]
    optimizer_SR_and_pattern_params = optim.Adam(params, lr=0.001)

    alpha = torch.tensor([1.0],device=device)
    alpha.requires_grad = True
    fusion_params = [{'params': alpha}]
    optimizer_fusion_params = optim.Adam(fusion_params, lr=0.01)

    input_SIM_raw_data = input_SIM_raw_data.to(device)

    input_SIM_raw_data_fft = unsample_process(image_size,input_SIM_raw_data,CTF,experimental_params.upsample)

    SR_image1 = first_step_optimization(SR_image_step1, input_SIM_raw_data_fft, optimizer_SR_and_pattern_params,estimated_SIM_pattern_without_m, estimated_modulation_factor, psf_reconstruction_conv, OTF)
    SR_image2 = second_step_optimization(SR_image_step2, input_SIM_raw_data_fft, optimizer_SR_and_pattern_params,
                             estimated_SIM_pattern_without_m, estimated_modulation_factor, OTF, psf_reconstruction_conv)

    SR = frequency_spectrum_fusion(SR_image1, SR_image2, estimated_pattern_parameters, optimizer_fusion_params, alpha,experimental_params)
    SR = SR / SR.max()
    SR_np = SR.cpu().squeeze().numpy()*255
    SSIM = SRimage_metrics.calculate_ssim(SR_np,HR)
    PSNR = SRimage_metrics.calculate_psnr_np(SR_np,HR)

    return SSIM,PSNR,SR


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
                torch.fft(input_SIM_raw_data_complex, 2)) * CTF.unsqueeze(2)

    return input_SIM_raw_data_fft

def first_step_optimization(SR_image,input_SIM_raw_data_fft,optimizer_SR_and_pattern_params,estimated_SIM_pattern_without_m,estimated_modulation_factor,psf_reconstruction_conv,OTF):
    ## step 1
    for epoch in range(num_epochs):

        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer_SR_and_pattern_params.zero_grad()
        temp_input_SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern_V1(estimated_SIM_pattern_without_m,
                                                                              estimated_modulation_factor,device)
        for i in range(3):
            sample_light_field = SR_image * temp_input_SIM_pattern[:, i, :, :]
            sample_light_field_complex = torch.stack([sample_light_field.squeeze(), torch.zeros_like(sample_light_field).squeeze()], 2)
            SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                torch.fft((sample_light_field_complex), 2)) * OTF.unsqueeze(2)
            mse_loss = criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:,i,:,:,:], 1)
            loss += mse_loss
        loss.backward()
        optimizer_SR_and_pattern_params.step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    SR_image_high_freq_filtered = forward_model.positive_propagate(SR_image.detach(), 1,
                                                                   psf_reconstruction_conv)
    SR_image1 = SR_image_high_freq_filtered
    common_utils.plot_single_tensor_image(SR_image1)

    return SR_image1

def second_step_optimization(SR_image_step2,input_SIM_raw_data_fft,optimizer_SR_and_pattern_params,estimated_SIM_pattern_without_m,estimated_modulation_factor,OTF,psf_reconstruction_conv):
    for epoch in range(num_epochs):
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer_SR_and_pattern_params.zero_grad()
        temp_input_SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern_V1(estimated_SIM_pattern_without_m,
                                                                                 estimated_modulation_factor, device)
        for i in range(temp_input_SIM_pattern.size()[1]):
            sample_light_field = SR_image_step2 * temp_input_SIM_pattern[:, i, :, :]
            sample_light_field_complex = torch.stack(
                [sample_light_field.squeeze(), torch.zeros_like(sample_light_field).squeeze()], 2)
            SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                torch.fft((sample_light_field_complex), 2)) * OTF.unsqueeze(2)
            mse_loss = criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:, i, :, :, :], 1)
            if i <= 2:
                loss += 1 / 3 * mse_loss
            else:
                loss += mse_loss
        loss.backward()
        optimizer_SR_and_pattern_params.step()
        with torch.no_grad():
            train_loss = loss.float()
        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    SR_image_high_freq_filtered = forward_model.positive_propagate(SR_image_step2.detach(), 1,
                                                                   psf_reconstruction_conv)
    SR_image2 = SR_image_high_freq_filtered
    common_utils.plot_single_tensor_image(SR_image2)

    return SR_image2

def frequency_spectrum_fusion(SR_image1,SR_image2,estimated_pattern_parameters,optimizer_fusion_params,alpha,experimental_params,mode = 'replace_LR'):
    SR_image_low_freq_complex = torch.stack(
        [SR_image1.squeeze(), torch.zeros_like(SR_image1).squeeze()], 2)
    SR_image_low_freq_fft = forward_model.torch_2d_fftshift(
        torch.fft((SR_image_low_freq_complex), 2))

    SR_image_high_freq_complex = torch.stack(
        [SR_image2.squeeze(), torch.zeros_like(SR_image2).squeeze()], 2)
    SR_image_high_freq_fft = forward_model.torch_2d_fftshift(
        torch.fft((SR_image_high_freq_complex), 2))

    CTF_bigger = experimental_params.CTF_form(fc_ratio=2.1).to(device)
    CTF_smaller = experimental_params.CTF_form(fc_ratio=1.9).to(device)
    CTF = experimental_params.CTF_form(fc_ratio=2).to(device)

    notch_filter_radius = 20
    notch_filter = processing_utils.notch_filter_generate(SR_image1, estimated_pattern_parameters,
                                                          notch_radius=notch_filter_radius)
    notch_filter_bigger = processing_utils.notch_filter_generate(SR_image1, estimated_pattern_parameters,
                                                                 notch_radius=notch_filter_radius + 1)
    notch_filter_smaller = processing_utils.notch_filter_generate(SR_image1, estimated_pattern_parameters,
                                                                  notch_radius=notch_filter_radius - 1)
    if mode == 'replace_LR':
        for i in range(num_epochs):
            optimizer_fusion_params.zero_grad()
            freq_intensity_LR = forward_model.complex_stack_to_intensity(SR_image_low_freq_fft * (CTF - CTF_smaller).unsqueeze(2))
            freq_intensity_HR = forward_model.complex_stack_to_intensity(SR_image_high_freq_fft * (CTF_bigger - CTF ).unsqueeze(2))
            loss = abs((freq_intensity_LR * alpha).sum() - freq_intensity_HR.sum())
            loss.backward()
            optimizer_fusion_params.step()
            with torch.no_grad():
                print('epoch: %d/%d, train_loss: %f,alpha : %f' % (i + 1, 100, loss,alpha))

        # result_fft = SR_image_low_freq_fft
        result_fft = SR_image_low_freq_fft * CTF.unsqueeze(2)*alpha  + SR_image_high_freq_fft* (1-CTF).unsqueeze(2)
        result = forward_model.complex_stack_to_intensity( torch.ifft(forward_model.torch_2d_ifftshift(result_fft),2) )

    elif mode == 'replace_peak':
        for i in range(num_epochs):
            optimizer_fusion_params.zero_grad()
            freq_intensity_LR = forward_model.complex_stack_to_intensity(
                SR_image_low_freq_fft * (notch_filter_smaller - notch_filter).unsqueeze(2))
            freq_intensity_HR = forward_model.complex_stack_to_intensity(
                SR_image_high_freq_fft * (notch_filter - notch_filter_bigger).unsqueeze(2))
            loss = abs((freq_intensity_LR * alpha).sum() - freq_intensity_HR.sum())
            loss.backward()
            optimizer_fusion_params.step()
            with torch.no_grad():
                print('epoch: %d/%d, train_loss: %f,alpha : %f' % (i + 1, 100, loss, alpha))
            # result_fft = SR_image_low_freq_fft
            # notch_filter = processing_utils.notch_filter_generate(SR_image1,estimated_pattern_parameters,notch_radius = 4)
            result_fft = SR_image_low_freq_fft * (1 - notch_filter).unsqueeze(
                2) * alpha + SR_image_high_freq_fft * notch_filter.unsqueeze(2)
            result = forward_model.complex_stack_to_intensity(
                torch.ifft(forward_model.torch_2d_ifftshift(result_fft), 2))
    else:
        raise Exception('please input correct mode')

    pattern_freq_radius = torch.norm(estimated_pattern_parameters[0, 0:2])
    result = processing_utils.notch_filter_for_all_vulnerable_point(result,
                                                                    estimated_pattern_parameters,
                                                                    filter_radius=pattern_freq_radius + 5).squeeze()
    # result = processing_utils.filter_for_computable_freq(result,estimated_pattern_parameters)
    result = processing_utils.notch_filter_single_direction(result,
                                                            torch.mean(estimated_pattern_parameters[0:3, :], dim=0))
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
    num_epochs = 1800

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    # criterion = nn.MSELoss()
    criterion = loss_functions.MSE_loss()
    num_raw_SIMdata, output_nc, num_downs = 2, 1, 5

    # compare_reconstruction_quality_using_different_input_frames(SIM_data_dataloader)
    start_time = time.time()

    input_id = 1
    data_id = 0
    for SIM_data, SIM_pattern in zip(SIM_data_dataloader, SIM_pattern_dataloader):
        # SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1

    SSIM, PSNR, best_SR = SR_reconstruction( SIM_data)
    if not best_SR.dim() == 4:
        best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    common_utils.save_image_tensor2pillow(best_SR, save_file_directory)
    end_time = time.time()

