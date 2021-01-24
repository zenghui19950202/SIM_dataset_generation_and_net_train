#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Using DIP(deep image prior)to achieve phase retrieval
# author：zenghui time:2021/1/13


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
from configparser import ConfigParser
from simulation_data_generation.generate_hologram_diffraction import hologram

# import self_supervised_learning_sr.estimate_SIM_pattern
import numpy as np


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')
    return device


def train(net, hologram_diffraction_loader, SIM_pattern_loader, net_input, criterion, num_epochs, device, lr=None,
          weight_decay=1e-5, opt_over='net'):
    print('training on', device)
    net = net.to(device)

    image_size = [net_input.size()[2], net_input.size()[3]]

    end_flag = 0

    input_id = 0
    data_id = 0
    for hologram_diffraction, _ in zip(hologram_diffraction_loader, SIM_pattern_loader):
        hologram_diffraction_intensity_raw_data = hologram_diffraction[0]
        if data_id == input_id:
            break
        data_id += 1

    # GT_phase_image = hologram_diffraction[1][:,:,:,0].squeeze()
    # estimated_phase_image += GT_phase_image.to(device)
    hologram_diffraction_intensity_raw_data = common_utils.pick_input_data(hologram_diffraction_intensity_raw_data)

    net_parameters = common_utils.get_params('net', net, downsampler=None, weight_decay=weight_decay)
    optimizer_net = optim.Adam(net_parameters, lr=lr)

    input_diffraction_intensity_raw_data = hologram_diffraction_intensity_raw_data.to(device)

    # load experimental parameters
    experimental_params = hologram(probability=1)
    data_num = experimental_params.data_num

    k = torch.tensor([experimental_params.wave_num], dtype=torch.float32).to(device)
    distance = torch.tensor([experimental_params.distance], dtype=torch.float32).to(device)
    lamda = torch.tensor([experimental_params.WaveLength], dtype=torch.float32).to(device)

    xx0, xx1, yy0, yy1 = experimental_params.xx0.to(device), experimental_params.xx1.to(
        device), experimental_params.yy0.to(device), experimental_params.yy1.to(device)


    noise = net_input.detach().clone()
    reg_noise_std = 0.03
    min_loss = 1e5

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, mode='min', factor=0.9, patience=100,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-5, eps=1e-08)

    for epoch in range(num_epochs):

        net.train()  # Switch to training mode

        net_input_noise = net_input + (noise.normal_() * reg_noise_std)
        net_input_noise = net_input_noise.to(device)

        optimizer_net.zero_grad()

        estimated_phase_image = net(net_input_noise)
        tv_loss = 1e-7 * loss_functions.tv_loss_calculate(estimated_phase_image)
        tv_loss=0
        estimated_phase_image = estimated_phase_image.squeeze()
        loss = tv_loss
        for i in range(data_num):
            d = (i + 1) * distance
            estimated_diffraction_hologram = forward_model.fresnel_propagate(estimated_phase_image, d, xx0, yy0,
                                                                             xx1,
                                                                             yy1, lamda, k)
            estimated_diffraction_hologram_intensity = pow(estimated_diffraction_hologram[:, :, 0], 2) + pow(
                estimated_diffraction_hologram[:, :, 1], 2)
            estimated_diffraction_hologram_intensity = estimated_diffraction_hologram_intensity / estimated_diffraction_hologram_intensity.max()
            mse_loss = criterion(estimated_diffraction_hologram_intensity,
                                 input_diffraction_intensity_raw_data[:, i, :, :],
                                 torch.ones_like(estimated_diffraction_hologram_intensity))
            loss += mse_loss
        loss.backward()
        optimizer_net.step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f  MSE_loss:%f  , tv_loss: %f' % (epoch + 1, num_epochs, train_loss, mse_loss,tv_loss ))
        if epoch == 999:  # safe checkpoint
            temp_loss = train_loss
            temp_net_state_dict = copy.deepcopy(net.state_dict())
            temp_optimizer_state_dict = copy.deepcopy(optimizer_net.state_dict())
            checkpoint_loss = train_loss

        if epoch > 1000:
            delta_loss = train_loss - temp_loss
            temp_loss = train_loss
            print('delta_loss_max:%f, loss_min: %f' % (3 * delta_loss.max(), min(train_loss, temp_loss)))
            if 3 * delta_loss > min(train_loss, temp_loss):
                net.load_state_dict(temp_net_state_dict)
                optimizer_net.load_state_dict(temp_optimizer_state_dict)
                temp_loss = checkpoint_loss
                end_flag += 1
                print('revert:True')
            elif epoch % 50 == 0:
                temp_net_state_dict = copy.deepcopy(net.state_dict())
                temp_optimizer_state_dict = copy.deepcopy(optimizer_net.state_dict())
                checkpoint_loss = train_loss

        if min_loss > train_loss:
            min_loss = train_loss
            best_estimated_phase_image = estimated_phase_image

        if (epoch + 1) % 1000 == 0:
            common_utils.plot_single_tensor_image(estimated_phase_image)
            # common_utils.plot_single_tensor_image(SR_image_high_freq_filtered)

        if end_flag > 5:
            break

    return train_loss, best_estimated_phase_image

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

def train_net():
    pass
if __name__ == '__main__':
    config = ConfigParser()
    if config.read('../configuration_hologram.ini') != []:
        config.read('../configuration_hologram.ini')
    elif config.read('configuration_hologram.ini') != []:
        config.read('configuration_hologram.ini')
    else:
        raise Exception('directory of configuration_hologram.ini error')

    train_net_parameters = {}
    train_directory_file = config.get('image_file', 'SourceFileDirectory') + '/SIMdata_SR_train.txt'
    save_file_directory = config.get('image_file', 'save_file_directory')
    data_num = config.getint('SIM_data_generation', 'data_num')
    image_size = config.getint('SIM_data_generation', 'image_size')
    opt_over = config.get('optimize_object', 'opt_over')

    SIM_data = SpeckleSIMDataLoad.SIM_data_load(train_directory_file, normalize=False, data_mode='only_raw_SIM_data')
    SIM_pattern = SpeckleSIMDataLoad.SIM_pattern_load(train_directory_file, normalize=False)
    # SIM_pattern = SIM_data_load(train_directory_file, normalize=False, data_mode='only_raw_SIM_data')

    SIM_data_dataloader = DataLoader(SIM_data, batch_size=1)
    SIM_pattern_dataloader = DataLoader(SIM_pattern, batch_size=1)

    random.seed(60)  # 设置随机种子
    # min_loss = 1e5
    num_epochs = 3000

    lr = 0.001
    weight_decay =1e-5

    device = try_gpu()
    # criterion = nn.MSELoss()
    criterion = loss_functions.MSE_loss()
    num_raw_SIMdata, output_nc, num_downs = 2, 1, 5
    # SIMnet = Unet_for_self_supervised.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,input_mode = 'only_input_SIM_images', use_dropout=False)
    # SIMnet = Networks_Unet_GAN.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,
    #                                                 input_mode='only_input_SIM_images', use_dropout=False)
    SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, 1, input_mode='input_all_images', LR_highway=False)
    # SIMnet = resnet_backbone_net._resnet('resnet34', resnet_backbone_net.BasicBlock, [1, 1, 1, 1], input_mode='only_input_SIM_images',
    #                             LR_highway=False, input_nc=num_raw_SIMdata, pretrained=False, progress=False, )
    # SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, 1, input_mode=data_input_mode, LR_highway=LR_highway_type)
    # SIMnet.apply(init_weights)
    # SIMnet = nn.Sequential()
    start_time = time.time()

    net_input = common_utils.get_noise(num_raw_SIMdata+1, 'noise', (image_size, image_size), var=0.1)
    net_input = net_input.to(device).detach()

    # net_input = SIM_data[0][1][:,:,0].squeeze()
    # net_input = torch.stack([net_input,net_input],0)
    # net_input = net_input.view(1,2,256,256)
    # net_input.requires_grad = True
    train_loss, best_hologram = train(SIMnet, SIM_data_dataloader, SIM_pattern_dataloader, net_input, criterion, num_epochs,
                                device, lr, weight_decay, opt_over)
    best_hologram = best_hologram.reshape([1, 1, image_size, image_size])
    common_utils.save_image_tensor2pillow(best_hologram, save_file_directory)

    best_hologram = best_hologram.reshape([1, 1, image_size, image_size])
    common_utils.save_image_tensor2pillow(best_hologram, save_file_directory)
    # SIMnet.to('cpu')
    end_time = time.time()
    # torch.save(SIMnet.state_dict(), file_directory + '/SIMnet.pkl')
    print(
        'avg train rmse: %f, learning_rate:%f, weight_decay: %f, time: %f '
        % (train_loss, lr, weight_decay, end_time - start_time))

    # a = SIM_train_dataset[0]
    # image = a[0]
    # image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    # print(SIMnet(image1))
