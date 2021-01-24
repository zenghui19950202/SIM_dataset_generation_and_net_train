#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/21
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
# from torch.fft import fftn


# import self_supervised_learning_sr.estimate_SIM_pattern
import numpy as np


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:5')
    else:
        device = torch.device('cpu')
    return device

def pick_input_images(SIM_raw_data,input_num):
    if input_num == 5:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data,[0, 1, 2, 3, 6])
    elif input_num == 6:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6, 4])
    elif input_num == 7:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6 ,4 ,7])
    elif input_num == 8:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 1, 2, 3, 6 ,4 ,7 ,5])
    elif input_num == 9:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data)
    elif input_num == 3:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data,[0, 3, 6])
    elif input_num == 4:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [0, 3, 6,1])
    elif input_num == 1:
        input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data, [2])

    return input_SIM_raw_data

def train(net, SIM_data_loader, SIM_pattern_loader, net_input, num_epochs, device, lr=None,
          weight_decay=1e-5, opt_over='net',input_num=9):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net = net.to(device)

    input_id = 1
    data_id = 0
    for SIM_data, SIM_pattern in zip(SIM_data_dataloader, SIM_pattern_loader):
        SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1

    LR_HR = SIM_data[1]
    HR = LR_HR[:,:,:,0]
    HR = HR / HR.max()
    HR = HR.squeeze().numpy()*255

    LR = LR_HR[:,:,:,1]
    LR = LR / LR.max()
    LR = LR.to(device)

    image_size = [SIM_raw_data.size()[2],SIM_raw_data.size()[3]]
    temp = funcs.SinusoidalPattern(probability=1,image_size = image_size[0])
    OTF = temp.OTF
    psf = temp.psf_form(OTF)

    psf_conv = funcs.psf_conv_generator(psf, device)

    CTF = temp.CTF_form(fc_ratio=2)
    CTF = CTF.to(device)
    # CTF = torch.ones_like(CTF)

    SSIM_list = []
    PSNR_list = []

    input_SIM_raw_data = pick_input_images(SIM_raw_data, input_num)
    input_SIM_pattern = pick_input_images(SIM_pattern, input_num)
    # input_SIM_pattern = common_utils.pick_input_data(SIM_pattern)
    # input_SIM_raw_data_normalized = processing_utils.pre_processing(input_SIM_raw_data)

    # SR_image = SIM_data[1][:,:,:,1]
    wide_field_image = torch.mean(input_SIM_raw_data[:, 0:3, :, :], dim=1)
    wide_field_image = wide_field_image / wide_field_image.max()
    # SR_image = forward_model.winier_deconvolution(wide_field_image,OTF)
    # SR_image = torch.rand_like(wide_field_image)
    experimental_params = funcs.SinusoidalPattern(probability=1, image_size=image_size[0])

    if experimental_params.upsample == True:
        up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        SR_image = up_sample(copy.deepcopy(wide_field_image).unsqueeze(0))
        OTF_upsmaple = experimental_params.OTF_upsmaple.to(device)
    else:
        SR_image = copy.deepcopy(wide_field_image)

    psf_radius = math.floor(psf.size()[0] / 2)

    mask = torch.zeros_like(input_SIM_raw_data, device=device)
    mask[:, :, psf_radius:-psf_radius, psf_radius:-psf_radius] = 1
    frequency_mask = 1 - temp.CTF_form(fc_ratio=0.2).to(device).unsqueeze(2)
    # frequency_mask = 1


    temp_input_SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels(
        input_SIM_raw_data)
    # illumination_intensity = estimated_pattern_parameters[:,4]
    # illumination_intensity = illumination_intensity.view(1,input_num,1,1)
    # temp_input_SIM_pattern = temp_input_SIM_pattern * illumination_intensity

    print(estimated_pattern_parameters)

    OTF_reconstruction = temp.OTF_form(fc_ratio=1 + temp.pattern_frequency_ratio)
    psf_reconstruction = temp.psf_form(OTF_reconstruction)
    psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction, device)

    experimental_parameters = SinusoidalPattern(probability=1,image_size =image_size[0])
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size[0], up_sample= experimental_parameters.upsample,grid_mode='pixel')

    estimated_modulation_factor = estimated_pattern_parameters[:,2].clone().detach()
    estimated_modulation_factor.requires_grad = True


    params = []
    SR_image = SR_image.to(device)
    SR_image.requires_grad = True
    params += [{'params': SR_image, 'weight_decay': weight_decay}]
    params += [{'params': estimated_modulation_factor}]
    optimizer_pattern_params = optim.Adam(params, lr=lr)


    input_SIM_raw_data = input_SIM_raw_data.to(device)
    input_SIM_pattern = input_SIM_pattern.to(device)
    temp_input_SIM_pattern = temp_input_SIM_pattern.to(device)
    # temp_input_SIM_pattern = input_SIM_pattern

    reg_noise_std = 0.03

    OTF = OTF.to(device)
    OTF_reconstruction = OTF_reconstruction.to(device)

    if experimental_parameters.upsample == True:
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

    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)

        net_input_noise = wide_field_image.normal_() * reg_noise_std
        net_input_noise = net_input_noise.to(device)

        # temp_input_SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern(temp_input_SIM_pattern,
        #                                                                       estimated_pattern_parameters,
        #                                                                       estimated_modulation_factor, xx, yy)
        optimizer_pattern_params.zero_grad()
        for i in range(temp_input_SIM_pattern.size()[1]):
            sample_light_field = SR_image * temp_input_SIM_pattern[:, i, :, :]
            sample_light_field_complex = torch.stack([sample_light_field.squeeze(), torch.zeros_like(sample_light_field).squeeze()], 2)

            if experimental_parameters.upsample == True:
                SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                    torch.fft((sample_light_field_complex), 2)) * OTF_upsmaple.unsqueeze(2)
            else:
                SIM_raw_data_fft_estimated = forward_model.torch_2d_fftshift(
                    torch.fft((sample_light_field_complex), 2)) * OTF.unsqueeze(2)
            mse_loss = criterion(SIM_raw_data_fft_estimated, input_SIM_raw_data_fft[:,i,:,:,:], frequency_mask,OTF = OTF, normalize=True, deconv = True)
            loss += mse_loss
            # tv_loss = 1e2 * loss_functions.tv_loss_calculate(SR_image.squeeze())
            # loss += tv_loss

        # if input_num == 3:
        #     wide_field_estimated = SR_image
        #     wide_field_estimated_complex = torch.stack(
        #         [wide_field_estimated.squeeze(), torch.zeros_like(wide_field_estimated).squeeze()], 2)
        #
        #     LR_complex = torch.stack(
        #         [LR.squeeze(), torch.zeros_like(LR).squeeze()], 2)
        #
        #     wide_field_fft_estimated = forward_model.torch_2d_fftshift(
        #             torch.fft((wide_field_estimated_complex), 2)) * OTF.unsqueeze(2)
        #
        #     wide_field_fft = forward_model.torch_2d_fftshift(
        #             torch.fft((LR_complex), 2))
        #
        #     wide_field_loss = criterion(wide_field_fft_estimated, wide_field_fft,1,OTF, True,deconv = True)
        #
        #     loss += wide_field_loss


        loss.backward()
        optimizer_pattern_params.step()

        with torch.no_grad():
            train_loss = loss.float()


        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))
        # print('epoch: %d/%d, train_loss: %f SSIM:%f , PSNR: %f' % (epoch + 1, num_epochs, train_loss, SSIM, PSNR))
        # SR_image_high_freq_filtered = forward_model.positive_propagate(SR_image.detach(), 1, psf_reconstruction_conv)
        # if min_loss > train_loss:
        #     min_loss = train_loss
        #     SR_image_high_freq_and_notch_filtered = processing_utils.notch_filter_for_all_vulnerable_point(SR_image_high_freq_filtered,
        #                                                                           estimated_pattern_parameters)
        #     best_SR = SR_image_high_freq_and_notch_filtered[psf_radius: -psf_radius, psf_radius: -psf_radius]
        #     # best_SR =SR_image_high_freq_filtered.squeeze()[psf_radius: -psf_radius, psf_radius: -psf_radius]
        if (epoch + 1) % num_epochs == 0:
            SR_image_high_freq_filtered = forward_model.positive_propagate(SR_image.detach(), 1,
                                                                           psf_reconstruction_conv)
            result = processing_utils.notch_filter_for_all_vulnerable_point(SR_image_high_freq_filtered, estimated_pattern_parameters).squeeze()
            # result = SR_image_high_freq_filtered.squeeze()
            common_utils.plot_single_tensor_image(result)

            SR = result / result.max()
            SR = SR.cpu().squeeze().numpy()*255
            SSIM = SRimage_metrics.calculate_ssim(SR,HR)
            PSNR = SRimage_metrics.calculate_psnr_np(SR,HR)

            # winier_result = forward_model.winier_deconvolution(processing_utils.notch_filter_for_all_vulnerable_point(SR_image_high_freq_filtered, estimated_pattern_parameters).squeeze(),OTF.cpu())
            # common_utils.plot_single_tensor_image(winier_result)

    return train_loss, result,SSIM,PSNR


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
    # SIMnet = Unet_for_self_supervised.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,input_mode = 'only_input_SIM_images', use_dropout=False)
    # SIMnet = Networks_Unet_GAN.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,
    #                                                 input_mode='only_input_SIM_images', use_dropout=False)
    SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, 1, input_mode='input_all_images', LR_highway=False)

    start_time = time.time()

    net_input = common_utils.get_noise(num_raw_SIMdata + 1, 'noise', (image_size, image_size), var=0.1)
    net_input = net_input.to(device).detach()

    train_loss, best_SR,SSIM,PSNR = train(SIMnet, SIM_data_dataloader, SIM_pattern_dataloader,
                                          net_input, num_epochs,device, lr, weight_decay, opt_over,input_num = 5)
    # best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    if not best_SR.dim() == 4:
        best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    common_utils.save_image_tensor2pillow(abs(best_SR), save_file_directory)
    torch.save(SSIM,save_file_directory+'SSIM')
    torch.save(PSNR,save_file_directory+'PSNR')
    # SIMnet.to('cpu')
    end_time = time.time()
    # torch.save(SIMnet.state_di== 4 and input_tensor.shape[0] == 1)ct(), file_directory + '/SIMnet.pkl')
    print('avg train rmse: %f, SSIM:%f, PSNR:%f,time: %f ' % (train_loss, SSIM,PSNR, end_time - start_time))
