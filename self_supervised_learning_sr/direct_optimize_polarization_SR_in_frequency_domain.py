#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Directly reconstruct the super resolution image and the polarization distribution using gradient descent
# author：zenghui time:2021/1/27


from parameter_estimation import *
from utils import *
# from models import *
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
from Deep_image_prior import denoise
from utils.image_processing import freq_domain_SIM_flower_filter
from utils.image_processing import generlized_wiener_filter
from self_supervised_learning_sr import PiFP_reconstruction

def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def reconstruction(SIM_data, SIM_pattern, input_num=5, image_show=True):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)

    LR_HR = SIM_data[1]
    SIM_raw_data = SIM_data[0]

    SIM_raw_data = common_utils.shuffle_image(SIM_raw_data,0)

    image_size = [SIM_raw_data.size()[2], SIM_raw_data.size()[3]]
    experimental_params = funcs.SinusoidalPattern(probability=1, image_size=image_size[0]) #
    polarization_ratio = calculate_polarization_ratio(SIM_raw_data, experimental_params) # estimate absorption efficiency from deconvovled wide-field images
    # polarization_ratio_regression = calculate_polarization_ratio_regression(SIM_raw_data, experimental_params)
    polarization_ratio_angle, h = calculate_polarization_direction(SIM_raw_data, experimental_params) # using polarization_direction_angle to calculate the absoprtion efficiency absorption_efficiency = 1 - 0.4 * torch.cos(2 * ( theta.reshape([1, 1, 3])-polarization_angle.unsqueeze(2)))
    # calculate_polar_approximation(polarization_ratio,polarization_ratio_angle,experimental_params)
    # polarization_ratio = polarization_ratio_regression
    # polarization_ratio = polarization_ratio_angle
    HR = LR_HR[:, :, :, 0]

    CTF = experimental_params.CTF_form(fc_ratio=1).to(device)

    if experimental_params.NumPhase == 2 and input_num == 6:
        input_num = 9
    input_SIM_raw_data = common_utils.input_data_pick(SIM_raw_data, input_num)
    input_polarization_ratio = common_utils.input_data_pick(polarization_ratio, input_num)
    input_SIM_pattern = common_utils.input_data_pick(SIM_pattern, input_num)
    """uncommenting codes of next line to use real value of polarization absorption efficiency for reconstruction"""
    # input_polarization_ratio = input_SIM_pattern
    # input_SIM_raw_data_normalized = processing_utils.pre_processing(input_SIM_raw_data)

    wide_field_image = torch.mean(input_SIM_raw_data[:, :, :, :], dim=1)
    wide_field_image = wide_field_image / wide_field_image.max()
    LR = torch.mean(SIM_raw_data[:, :, :, :], dim=1).unsqueeze(0)
    # SR_image = forward_model.winier_deconvolution(wide_field_image,OTF)
    # SR_image = torch.rand_like(wide_field_image)

    if experimental_params.upsample == True:
        up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        SR_image = up_sample(copy.deepcopy(wide_field_image).unsqueeze(0))
        HR = up_sample(HR.unsqueeze(0))
        LR = up_sample(LR)
        h = up_sample(h.unsqueeze(0).unsqueeze(0))
        # SR_image = torch.zeros_like(SR_image)
    else:
        SR_image = copy.deepcopy(wide_field_image).unsqueeze(0)
    HR = HR / HR.max()
    HR = HR.squeeze().numpy() * 255
    #
    # temp_input_SIM_pattern, estimated_pattern_parameters, estimated_SIM_pattern_without_m = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
    #     input_SIM_raw_data, experimental_params)

    temp_input_SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_TIRF_image(input_SIM_raw_data, SIM_raw_data, experimental_params)

    print(estimated_pattern_parameters)
    # for i in range(temp_input_SIM_pattern.size()[1]):
    #     common_utils.save_image_tensor2pillow(temp_input_SIM_pattern[0,i,:,:].unsqueeze(0).unsqueeze(0), save_file_directory)
    #     time.sleep(3)

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
    optimizer_SR_and_polarization = optim.Adam(params, lr=lr)


    input_SIM_raw_data = input_SIM_raw_data.to(device)

    input_SIM_raw_data, input_SIM_raw_data_fft = unsample_process(image_size, input_SIM_raw_data, CTF, experimental_params.upsample)

    if experimental_params.upsample:
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        input_polarization_ratio = upsample(input_polarization_ratio)
        # input_polarization_ratio = upsample(input_SIM_pattern)
    input_polarization_ratio = input_polarization_ratio.to(device)

    reconstruction_start = time.time()
    SR_result = SR_reconstruction(SR_image, input_SIM_raw_data_fft, optimizer_SR_and_polarization,
                                  input_polarization_ratio,
                                  temp_input_SIM_pattern, estimated_pattern_parameters, experimental_params,input_SIM_raw_data,
                                  image_show=image_show,LR_image = LR)

    """"
    PiFP reconstruction
    """
    "integrate the SIM_pattern and polar_absorb into one variable"

    # input_SIM_raw_data = input_SIM_raw_data.detach().cpu().squeeze()
    # input_SIM_pattern = temp_input_SIM_pattern.detach().cpu().squeeze()
    # polarization_ratio = input_polarization_ratio.detach().cpu().squeeze()
    # input_SIM_pattern *= polarization_ratio
    #
    # PiFP_SR_result = PiFP_reconstruction.PiFP_recon(input_SIM_raw_data, input_SIM_pattern, SR_result.cpu().squeeze() , experimental_params)

    reconstruction_end = time.time()
    print('reconstruction_time:%f' % (reconstruction_end-reconstruction_start))

    # index = SR_result > SR_result.mean() * 5
    # SR_result[index] = 0
    common_utils.plot_single_tensor_image(h*255)
    SR = SR_result / SR_result.max()
    nn.AvgPool2d
    s = 0.6 * torch.ones_like(h)
    v = SR
    RGB = HSV2BGR(torch.stack([h.squeeze(), s.squeeze(), v], 2))
    # plt.imshow(RGB)
    # plt.show()
    crop_size = 20
    crop_size = 0
    if crop_size != 0:
        SR = SR[crop_size:-crop_size, crop_size:-crop_size]
        HR = HR[crop_size:-crop_size, crop_size:-crop_size]
    SR = SR / SR.max()
    SR_np = SR.cpu().squeeze().detach().numpy() * 255

    SSIM = SRimage_metrics.calculate_ssim(SR_np, HR)
    PSNR = SRimage_metrics.calculate_psnr_np(SR_np, HR)
    # SSIM = 1
    # PSNR = 1
    LR = LR/LR.max()
    SR_plus = abs( (SR - LR) + SR )
    SR_plus = SR_plus / SR_plus.max()
    # common_utils.plot_single_tensor_image(SR_plus)

    return SSIM, PSNR, abs(SR)  # (1+torch.cos(polarization_direction))


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

def SR_reconstruction(SR_image, input_SIM_raw_data_fft, optimizer_SR_and_polarization,
                      polarization_ratio, input_SIM_pattern, estimated_pattern_parameters, experimental_params,input_SIM_raw_data,
                      image_show=True,LR_image = None):
    device = SR_image.device
    if experimental_params.upsample == True:
        OTF = experimental_params.OTF_upsmaple.to(device)
        CTF = experimental_params.CTF_upsmaple.to(device)
        fx, fy = experimental_params.fx_upsmaple, experimental_params.fy_upsmaple
    else:
        OTF = experimental_params.OTF.to(device)
        CTF = experimental_params.CTF.to(device)
        fx, fy = experimental_params.fx, experimental_params.fy
    OTF_extended = experimental_params.OTF_form(fc_ratio=3, upsample=experimental_params.upsample)
    fr = pow(fx, 2) + pow(fy, 2)
    sigma = experimental_params.f_cutoff / 4
    high_pass_filter = 1 - torch.exp(-fr / sigma ** 2)
    high_pass_filter = high_pass_filter.unsqueeze(2).to(device)
    high_pass_filter = 1
    LR_image = LR_image.to(device)
    # common_utils.plot_single_tensor_image(high_pass_filter)

    loss_list = iterative_recon_one_shot(optimizer_SR_and_polarization, estimated_pattern_parameters, SR_image, input_SIM_pattern,
                             polarization_ratio, OTF, input_SIM_raw_data_fft,input_SIM_raw_data)

    # loss_list = iterative_recon_three_shot(optimizer_SR_and_polarization, estimated_pattern_parameters, SR_image, input_SIM_pattern,
    #                          polarization_ratio, OTF, input_SIM_raw_data_fft)

    # loss_list,SR_image = iterative_recon_three_shot_individual(SR_image, input_SIM_pattern,
    #                                       polarization_ratio, OTF, input_SIM_raw_data_fft)


    loss_np = np.array(loss_list)
    # common_utils.save_loss_npy(loss_np,SR_result_save_directory)
    # loss_load = np.load(loss_save_direc+'.npy')
    epoch_list = [x for x in range(1,num_epochs+1,1)]
    plt.plot(epoch_list,loss_np,c='red')
    plt.show()

    if image_show == True:
        result = SR_image.squeeze()
        result = processing_utils.notch_filter_for_all_vulnerable_point(SR_image, estimated_pattern_parameters,
                                                                        experimental_params,attenuate_factor = 2).squeeze().detach().cpu()
        spatial_freq = torch.sqrt(estimated_pattern_parameters[0,0]**2 + estimated_pattern_parameters[0,1]**2).detach().cpu()*experimental_params.delta_fx
        apodization_func = experimental_params.apodization_function_generator(experimental_params.f_cutoff+spatial_freq,upsample = experimental_params.upsample)

        # result = experimental_params.OTF_Filter(result.detach().cpu(), apodization_func.detach().cpu())
        # result = freq_domain_SIM_flower_filter(result.detach().cpu(), estimated_pattern_parameters,CTF)
        # result = generlized_wiener_filter(result.detach().cpu(),estimated_pattern_parameters)
        common_utils.plot_single_tensor_image(result)

    return result.cpu().detach()

def iterative_recon_one_shot(optimizer_SR_and_polarization,estimated_pattern_parameters,SR_image,input_SIM_pattern,polarization_ratio,OTF,input_SIM_raw_data_fft,input_SIM_raw_data):
    loss_list = []

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

            # SIM_raw_data_estimated = forward_model.complex_stack_to_intensity(torch.ifft(forward_model.torch_2d_ifftshift(SIM_raw_data_fft_estimated),2) )
            # mse_loss = criterion(input_SIM_raw_data[:, i, :, :], SIM_raw_data_estimated,1, OTF, normalize=False, deconv=False)

            mse_loss = criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:, i, :, :, :], 1,
                                 OTF, normalize=False, deconv=False)
            loss += mse_loss
        # loss += loss_functions.tv_loss_calculate(abs(SR_image),1)
        # loss += 5 * torch.norm(abs(SR_image),1)

        # for param_group in optimizer_SR_and_polarization.param_groups:
        #     lr = lr
        #     param_group['lr'] = lr
        loss.backward()
        optimizer_SR_and_polarization.step()

        with torch.no_grad():
            train_loss = loss.float()
            loss_list.append(loss.detach().cpu().numpy())
        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    return loss_list

def iterative_recon_three_shot(optimizer_SR_and_polarization, estimated_pattern_parameters, SR_image, input_SIM_pattern,
                             polarization_ratio, OTF, input_SIM_raw_data_fft):

    loss_list = []
    loss = torch.tensor([0.0], dtype=torch.float32, device=device)
    optimizer_SR_and_polarization.zero_grad()

    if input_num == 6:
        num_one_dir = 2
    elif input_num == 9:
        num_one_dir = 3

    for epoch in range(num_epochs):

        for i in range(estimated_pattern_parameters.size()[0]):
            theta = torch.atan(estimated_pattern_parameters[i, 1] / estimated_pattern_parameters[i, 0])
            # sample_light_field = SR_image * input_SIM_pattern[:, i, :, :] * (1 + 0.6 * torch.cos(2 * theta - 2 * polarization_direction))
            # sample_light_field = SR_image * input_SIM_pattern[:, i, :, :] * polarization_ratio[:, i, :, :]
            sample_light_field = SR_image * input_SIM_pattern[:, i, :, :]
            sample_light_field_complex = torch.stack(
                [sample_light_field.squeeze(), torch.zeros_like(sample_light_field).squeeze()], 2)
            SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                torch.fft((sample_light_field_complex), 2)) * OTF.unsqueeze(2)

            mse_loss = criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:, i, :, :, :], 1,
                                 OTF,
                                 normalize=False, deconv=False)
            loss += mse_loss
            if i % num_one_dir == 1:
                loss += loss_functions.tv_loss_calculate(abs(SR_image), 1)
                loss += 5 * torch.norm(abs(SR_image), 1)
                loss.backward()
                optimizer_SR_and_polarization.step()
                loss = torch.tensor([0.0], dtype=torch.float32, device=device)
                optimizer_SR_and_polarization.zero_grad()

        with torch.no_grad():
            train_loss = loss.float()
            loss_list.append(loss.detach().cpu().numpy())
        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    return loss_list

def iterative_recon_three_shot_individual(SR_image, input_SIM_pattern,
                             polarization_ratio, OTF, input_SIM_raw_data_fft):

    loss_list = []

    SR_image0 = copy.deepcopy(SR_image)
    SR_image1 = copy.deepcopy(SR_image)
    SR_image2 = copy.deepcopy(SR_image)

    SR_image0.requires_grad, SR_image1.requires_grad, SR_image2.requires_grad = True, True, True

    params0 = [{'params': SR_image0, 'weight_decay': weight_decay}]
    params1 = [{'params': SR_image1, 'weight_decay': weight_decay}]
    params2 = [{'params': SR_image2, 'weight_decay': weight_decay}]
    optimizer_pattern_params0 = optim.Adam(params0, lr=lr)
    optimizer_pattern_params1 = optim.Adam(params1, lr=lr)
    optimizer_pattern_params2 = optim.Adam(params2, lr=lr)

    SR_image_list = [SR_image0, SR_image1, SR_image2]

    optimizer = [optimizer_pattern_params0, optimizer_pattern_params1, optimizer_pattern_params2]

    if input_num == 6:
        num_phase = 2
    elif input_num == 9:
        num_phase = 3

    for epoch in range(num_epochs):

        i = int(epoch * 3 / num_epochs)
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer[i].zero_grad()
        SR_image_iterative = SR_image_list[i]
        for j in range(num_phase):
            num_SIM_data = i * num_phase + j
            # sample_light_field = SR_image_interative * input_SIM_pattern[:, num_SIM_data , :, :] * polarization_ratio[:, num_SIM_data, :, :]
            sample_light_field = SR_image_iterative * input_SIM_pattern[:, num_SIM_data, :, :]

            sample_light_field_complex = torch.stack(
                [sample_light_field.squeeze(), torch.zeros_like(sample_light_field).squeeze()], 2)
            SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                torch.fft((sample_light_field_complex), 2)) * OTF.unsqueeze(2)
            loss += criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:, num_SIM_data, :, :, :], 1, OTF, normalize=False, deconv=False)


        # loss += loss_functions.tv_loss_calculate(abs(SR_image_iterative), 1)
        # loss += 5 * torch.norm(abs(SR_image_interative), 1)
        loss.backward()
        optimizer[i].step()

        with torch.no_grad():
            train_loss = loss.float()
            loss_list.append(loss.detach().cpu().numpy())
        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    SR_image = (SR_image0 + SR_image1 + SR_image2)/3

    return loss_list, SR_image

def unsample_process(image_size, input_SIM_raw_data, CTF, upsample_flag=False):
    device = CTF.device
    input_num = input_SIM_raw_data.size()[1]
    up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    if upsample_flag == True:
        SIM_raw_data = up_sample(copy.deepcopy(input_SIM_raw_data))
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
        SIM_raw_data = copy.deepcopy(input_SIM_raw_data)
        input_SIM_raw_data_fft = torch.zeros([1, input_num, image_size[0], image_size[1], 2], device=device)
        for i in range(input_SIM_raw_data.size()[1]):
            input_SIM_raw_data_complex = torch.stack(
                [input_SIM_raw_data[:, i, :, :].squeeze(), torch.zeros_like(input_SIM_raw_data[:, i, :, :]).squeeze()],
                2)

            input_SIM_raw_data_fft[:, i, :, :, :] = forward_model.torch_2d_fftshift(
                torch.fft(input_SIM_raw_data_complex, 2)) * CTF.unsqueeze(2)

    return SIM_raw_data, input_SIM_raw_data_fft

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
        'learning_rate': [0.01],
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
    num_epochs =  600

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

    SR_result_save_directory = save_file_directory + 'lr_'+str(lr) + '_' + 'epoch_' + str(num_epochs)

    input_num = 3
    SSIM, PSNR, best_SR = reconstruction(SIM_data, SIM_pattern, input_num=input_num)

    # denoise_SIM_SR = denoise.denoise_DIP(best_SR.float())
    # common_utils.plot_single_tensor_image(denoise_SIM_SR)

    if best_SR.dim() == 3:
        best_SR = best_SR.reshape([1, best_SR.size()[0], best_SR.size()[1], 3])
    elif best_SR.dim() == 2:
        best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])

    common_utils.save_image_tensor2pillow(best_SR, SR_result_save_directory)
    end_time = time.time()

    print(' SSIM:%f, PSNR:%f,time: %f ' % (SSIM, PSNR, end_time - start_time))
