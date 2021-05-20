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
from parameter_estimation.estimate_polarizaion import *
import os
import numpy as np
import matplotlib.pyplot as plt


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:6')
    else:
        device = torch.device('cpu')
    return device


def compare_reconstruction_quality_using_different_input_frames(SIM_data_dataloader, image_show=False):
    data_num = 0
    for SIM_data in SIM_data_dataloader:
        SIM_raw_data = SIM_data[0]
        SSIM_of_diff_input_num = torch.zeros(5, 1)
        PSNR_of_diff_input_num = torch.zeros(5, 1)
        for i in range(5):
            SSIM_of_diff_input_num[i, 0], PSNR_of_diff_input_num[i, 0], SR = SR_reconstruction(SIM_data,
                                                                                               input_num=i + 5,
                                                                                               image_show=image_show)
        if data_num == 0:
            SSIM = SSIM_of_diff_input_num
            PSNR = PSNR_of_diff_input_num
        else:
            SSIM = torch.cat([SSIM, SSIM_of_diff_input_num], 1)
            PSNR = torch.cat([PSNR, PSNR_of_diff_input_num], 1)
        data_num += 1
        if data_num > 30:
            break

    SSIM_mean = torch.mean(SSIM, 1).numpy()
    SSIM_std = torch.std(SSIM, 1).numpy()

    PSNR_mean = torch.mean(PSNR, 1).numpy()
    PSNR_std = torch.std(PSNR, 1).numpy()

    np.save(save_file_directory + "SSIM.npy", SSIM.numpy())
    np.save(save_file_directory + "PSNR.npy", PSNR.numpy())

    index = np.arange(5)
    total_width, n = 0.4, 2
    width = total_width / n

    plt.title('A Bar Chart')
    plt.bar(index, SSIM_mean, width=width, yerr=SSIM_std, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7,
            label='SSIM', color='#583d72')
    plt.legend(loc=2)
    plt.savefig(save_file_directory + 'SSIM_bar.eps', dpi=600, format='eps')
    plt.show()
    plt.bar(index - width, PSNR_mean, width=width, yerr=PSNR_std, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7,
            label='PSNR', color='#9f5f80')
    plt.xticks(index + 0.2, ['5', '6', '7', '8', '9'])
    plt.legend(loc=2)
    plt.grid(linestyle='--', c='#bbbbbb')
    plt.savefig(save_file_directory + 'PSNR_bar.eps', dpi=600, format='eps')
    plt.show()

def calculate_polar_approximation(polarization_ratio_direct,polarization_ratio_angle_estimated,experimental_params):
    # polarization_ratio 是直接除法近似得到， polarization_ratio_estimated是先计算偏振角再计算得到
    polarization_ratio_load = torch.zeros_like(polarization_ratio_direct)
    SSIM_angle,PSNR_angle,mse_loss_angle  = 0,0,0
    SSIM_direct,PSNR_direct,mse_loss_direct  = 0,0,0
    SSIM_direct_and_angle,PSNR_direct_and_angle =0,0
    mse = nn.MSELoss()
    file_name = save_file_directory + '/SIMdata_SR_train/0absorption_efficiency.pt'
    for i in range(9):
        polar_estimated_direct = polarization_ratio_direct[0, i, :, :]
        polar_estimated_angle = polarization_ratio_angle_estimated[0, i, :, :]
        polar_estimated_angle /= polar_estimated_angle.max()

        polar_estimated_direct /= polar_estimated_direct.max()

        polar_estimated_angle_np = polar_estimated_angle.cpu().squeeze().detach().numpy() * 255
        polar_estimated_direct_np = polar_estimated_direct.cpu().squeeze().detach().numpy() * 255

        if os.path.exists(file_name):
            polarization_ratio_load[0, i, :, :] = torch.load(
                save_file_directory + '/SIMdata_SR_train/' + str(i) + 'absorption_efficiency.pt')

            polar_GT = experimental_params.OTF_Filter(polarization_ratio_load[0, i, :, :], experimental_params.OTF)
            polar_GT = polar_GT / polar_GT.max()
            polar_GT_np = polar_GT.cpu().squeeze().detach().numpy() * 255


            SSIM_direct += SRimage_metrics.calculate_ssim(polar_estimated_direct_np, polar_GT_np)
            PSNR_direct += SRimage_metrics.calculate_psnr_np(polar_estimated_direct_np, polar_GT_np)
            mse_loss_direct+=mse(polar_GT, polar_estimated_direct)

            SSIM_angle += SRimage_metrics.calculate_ssim(polar_estimated_angle_np, polar_GT_np)
            PSNR_angle += SRimage_metrics.calculate_psnr_np(polar_estimated_angle_np, polar_GT_np)
            mse_loss_angle += mse(polar_GT, polar_estimated_angle)
    # polarization_ratio = polarization_ratio_load

        else:
            SSIM_direct_and_angle+= SRimage_metrics.calculate_ssim(polar_estimated_direct_np, polar_estimated_angle_np)
            PSNR_direct_and_angle += SRimage_metrics.calculate_psnr_np(polar_estimated_direct_np, polar_estimated_angle_np)
    if os.path.exists(file_name):
        SSIM_direct /= 9
        PSNR_direct /= 9
        mse_loss_direct /= 9
        SSIM_angle /= 9
        PSNR_angle /= 9
        mse_loss_angle /= 9
        print(' polarization_ratio: SSIM_direct:%f, PSNR_direct:%f ' % (SSIM_direct, PSNR_direct))
        print(' polarization_ratio: SSIM_angle:%f, PSNR_angle:%f ' % (SSIM_angle, PSNR_angle))
    else:
        SSIM_direct_and_angle/= 9
        PSNR_direct_and_angle/=9
        print(' No GT data, polarization_ratio: SSIM_direct_and_angle:%f, PSNR_direct_and_angle:%f ' % (SSIM_direct_and_angle, PSNR_direct_and_angle))
def reconstruction(SIM_data, SIM_pattern, input_num=5, image_show=True):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)

    LR_HR = SIM_data[1]
    SIM_raw_data = SIM_data[0]

    image_size = [SIM_raw_data.size()[2], SIM_raw_data.size()[3]]
    experimental_params = funcs.SinusoidalPattern(probability=1, image_size=image_size[0])
    polarization_ratio = calculate_polarization_ratio(SIM_raw_data, experimental_params)
    # polarization_ratio_regression = calculate_polarization_ratio_regression(SIM_raw_data, experimental_params)
    polarization_ratio_angle, h = calculate_polarization_direction(SIM_raw_data, experimental_params)
    # calculate_polar_approximation(polarization_ratio,polarization_ratio_angle,experimental_params)
    # polarization_ratio = polarization_ratio_regression
    # polarization_ratio = polarization_ratio_angle
    HR = LR_HR[:, :, :, 0]

    CTF = experimental_params.CTF_form(fc_ratio=1).to(device)

    if experimental_params.NumPhase ==2:
        input_num = 9
    input_SIM_raw_data = common_utils.input_data_pick(SIM_raw_data, input_num)
    input_polarization_ratio = common_utils.input_data_pick(polarization_ratio, input_num)
    input_SIM_pattern = common_utils.input_data_pick(SIM_pattern, input_num)
    # input_polarization_ratio = input_SIM_pattern
    # input_SIM_raw_data_normalized = processing_utils.pre_processing(input_SIM_raw_data)

    wide_field_image = torch.mean(input_SIM_raw_data[:, 0:3, :, :], dim=1)
    wide_field_image = wide_field_image / wide_field_image.max()
    # SR_image = forward_model.winier_deconvolution(wide_field_image,OTF)
    # SR_image = torch.rand_like(wide_field_image)

    if experimental_params.upsample == True:
        up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        SR_image = up_sample(copy.deepcopy(wide_field_image).unsqueeze(0))
        HR = up_sample(HR.unsqueeze(0))
        h = up_sample(h.unsqueeze(0).unsqueeze(0))
    else:
        SR_image = copy.deepcopy(wide_field_image).unsqueeze(0)
    HR = HR / HR.max()
    HR = HR.squeeze().numpy() * 255

    temp_input_SIM_pattern, estimated_pattern_parameters, estimated_SIM_pattern_without_m = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
        input_SIM_raw_data, experimental_params)

    print(estimated_pattern_parameters)
    temp_input_SIM_pattern = temp_input_SIM_pattern.to(device)
    # temp_input_SIM_pattern = input_SIM_pattern.to(device)
    estimated_pattern_parameters = estimated_pattern_parameters.to(device)
    estimated_modulation_factor = estimated_pattern_parameters[:, 2].clone().detach().to(device)
    estimated_modulation_factor.requires_grad = True

    params = []
    SR_image = SR_image.to(device)
    SR_image.requires_grad = True

    # polarization_ratio = torch.tensor([0.0]).to(device)
    # polarization_ratio.requires_grad = True

    params += [{'params': SR_image, 'weight_decay': weight_decay}]
    # params += [{'params': estimated_modulation_factor}]
    # optimizer_SR_and_polarization = optim.Adam(params, lr=0.001)
    optimizer_SR_and_polarization = optim.Adam(params, lr=0.005)


    input_SIM_raw_data = input_SIM_raw_data.to(device)

    input_SIM_raw_data_fft = unsample_process(image_size, input_SIM_raw_data, CTF, experimental_params.upsample)

    if experimental_params.upsample:
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        input_polarization_ratio = upsample(input_polarization_ratio)
    input_polarization_ratio = input_polarization_ratio.to(device)

    SR_result = SR_reconstruction(SR_image, input_SIM_raw_data_fft, optimizer_SR_and_polarization,
                                  input_polarization_ratio,
                                  temp_input_SIM_pattern, estimated_pattern_parameters, experimental_params,
                                  image_show=image_show)
    SR = SR_result / SR_result.max()
    nn.AvgPool2d
    s = 0.6 * torch.ones_like(h)
    v = SR
    RGB = HSV2BGR(torch.stack([h.squeeze(), s.squeeze(), v], 2))
    plt.imshow(RGB)
    plt.show()

    SR_np = SR.cpu().squeeze().detach().numpy() * 255
    SSIM = SRimage_metrics.calculate_ssim(SR_np, HR)
    PSNR = SRimage_metrics.calculate_psnr_np(SR_np, HR)
    # SSIM = 1
    # PSNR = 1

    return SSIM, PSNR, RGB  # (1+torch.cos(polarization_direction))


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def unsample_process(image_size, input_SIM_raw_data, CTF, upsample_flag=False):
    device = CTF.device
    input_num = input_SIM_raw_data.size()[1]
    if upsample_flag == True:
        if image_size[0] % 2 == 0:
            padding_size = int(image_size[0] / 2)
            ZeroPad = nn.ZeroPad2d(padding=(padding_size, padding_size, padding_size, padding_size))

        else:
            padding_size = int(image_size[0] / 2)
            ZeroPad = nn.ZeroPad2d(padding=(padding_size + 1, padding_size, padding_size + 1, padding_size))

        input_SIM_raw_data_fft = torch.zeros([1, input_num, image_size[0] * 2, image_size[1] * 2, 2], device=device)
        for i in range(input_SIM_raw_data.size()[1]):
            input_SIM_raw_data_complex = torch.stack(
                [input_SIM_raw_data[:, i, :, :].squeeze(), torch.zeros_like(input_SIM_raw_data[:, i, :, :]).squeeze()],
                2)
            SIM_raw_data_fft = forward_model.torch_2d_fftshift(
                torch.fft(input_SIM_raw_data_complex, 2)) * CTF.unsqueeze(2)

            input_SIM_raw_data_fft[:, i, :, :, :] = torch.stack(
                [ZeroPad(SIM_raw_data_fft[:, :, 0]), ZeroPad(SIM_raw_data_fft[:, :, 1])], 2)
    else:
        input_SIM_raw_data_fft = torch.zeros([1, input_num, image_size[0], image_size[1], 2], device=device)
        for i in range(input_SIM_raw_data.size()[1]):
            input_SIM_raw_data_complex = torch.stack(
                [input_SIM_raw_data[:, i, :, :].squeeze(), torch.zeros_like(input_SIM_raw_data[:, i, :, :]).squeeze()],
                2)

            input_SIM_raw_data_fft[:, i, :, :, :] = forward_model.torch_2d_fftshift(
                torch.fft(input_SIM_raw_data_complex, 2)) * CTF.unsqueeze(2)

    return input_SIM_raw_data_fft


def SR_reconstruction(SR_image, input_SIM_raw_data_fft, optimizer_SR_and_polarization,
                      polarization_ratio, input_SIM_pattern, estimated_pattern_parameters, experimental_params,
                      image_show=True):
    device = SR_image.device
    if experimental_params.upsample == True:
        OTF = experimental_params.OTF_upsmaple.to(device)
        fx, fy = experimental_params.fx_upsmaple, experimental_params.fy_upsmaple
    else:
        OTF = experimental_params.OTF.to(device)
        fx, fy = experimental_params.fx, experimental_params.fy
    OTF_extended = experimental_params.OTF_form(fc_ratio=3, upsample=experimental_params.upsample)
    fr = pow(fx, 2) + pow(fy, 2)
    sigma = experimental_params.f_cutoff / 4
    high_pass_filter = 1 - torch.exp(-fr / sigma ** 2)
    high_pass_filter = high_pass_filter.unsqueeze(2).to(device)
    high_pass_filter = 1
    # common_utils.plot_single_tensor_image(high_pass_filter)

    for epoch in range(num_epochs):
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer_SR_and_polarization.zero_grad()
        for i in range(estimated_pattern_parameters.size()[0]):
            theta = torch.atan(estimated_pattern_parameters[i, 1] / estimated_pattern_parameters[i, 0])
            # sample_light_field = SR_image * input_SIM_pattern[:, i, :, :] * (1 + 0.6 * torch.cos(2 * theta - 2 * polarization_direction))
            sample_light_field = SR_image * input_SIM_pattern[:, i, :, :] * polarization_ratio[:, i, :, :]
            # sample_light_field = SR_image * input_SIM_pattern[:, i, :, :]
            sample_light_field_complex = torch.stack(
                [sample_light_field.squeeze(), torch.zeros_like(sample_light_field).squeeze()], 2)
            SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                torch.fft((sample_light_field_complex), 2)) * OTF.unsqueeze(2)

            mse_loss = criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:, i, :, :, :], high_pass_filter,
                                 OTF,
                                 normalize=False, deconv=False)
            loss += mse_loss
        # loss+=   loss_functions.tv_loss_calculate(abs(SR_image))
        # for param_group in optimizer_SR_and_polarization.param_groups:
        #     lr = lr
        #     param_group['lr'] = lr
        loss.backward()
        optimizer_SR_and_polarization.step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    if image_show == True:
        result = SR_image.squeeze()
        result = processing_utils.notch_filter_for_all_vulnerable_point(SR_image, estimated_pattern_parameters,
                                                                        experimental_params).squeeze().detach().cpu()
        result = experimental_params.OTF_Filter(result.detach().cpu(), OTF_extended.detach().cpu())
        common_utils.plot_single_tensor_image(result)

    return result.cpu().detach()


# def polarization_reconstruction(SIM_data, input_num=4, SR):

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

    SIM_data = SpeckleSIMDataLoad.SIM_data_load(train_directory_file, normalize=True, data_mode='only_raw_SIM_data')
    SIM_pattern = SpeckleSIMDataLoad.SIM_pattern_load(train_directory_file, normalize=True)
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

    SSIM, PSNR, best_SR = reconstruction(SIM_data, SIM_pattern, input_num=6)
    if best_SR.dim() == 3:
        best_SR = best_SR.reshape([1, best_SR.size()[0], best_SR.size()[1], 3])
    elif best_SR.dim() == 2:
        best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    common_utils.save_image_tensor2pillow(best_SR, save_file_directory)
    end_time = time.time()

    print(' SSIM:%f, PSNR:%f,time: %f ' % (SSIM, PSNR, end_time - start_time))
