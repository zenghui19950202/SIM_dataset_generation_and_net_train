#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2021/1/14

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/9
from utils import *
import torch
import torch.optim as optim
import torch.nn as nn
import random
import time
import copy
import math
from simulation_data_generation import fuctions_for_generate_pattern as funcs
from Deep_image_prior.Unet_for_denoise import  UNet
import cv2
import numpy as np


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def train(net, SIM_SR, Perturbation, criterion, num_epochs, device, experimental_parameters, lr=None,
          weight_decay=1e-5):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net = net.to(device)
    SIM_SR = SIM_SR.to(device)
    noise = Perturbation.detach().clone()
    reg_noise_std = 0.03

    end_flag = 0

    net_parameters = common_utils.get_params('net', net, downsampler=None,
                                                  weight_decay=weight_decay)
    optimizer_net = optim.Adam(net_parameters, lr=lr)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, mode='min', factor=0.9, patience=100,
    #                                                        verbose=False, threshold=0.0001, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=1e-5, eps=1e-08)

    net = net.to(device)

    count_epoch = 1500
    switch_flag = 1
    min_loss = 1e5
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        net_input_noise = Perturbation + (noise.normal_() * reg_noise_std)
        net_input_noise = net_input_noise.to(device)

        optimizer_net.zero_grad()
        denoise_image = net(net_input_noise)
        # denoise_image = abs(denoise_image)
        mse_loss = criterion(SIM_SR, denoise_image)
        loss = mse_loss

        loss.backward()
        optimizer_net.step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f ' % (epoch + 1, num_epochs, train_loss ))
        # SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern(input_SIM_raw_data,estimated_pattern_parameters,delta_pattern_params,xx,yy)
        # print(delta_pattern_params)
        if epoch == count_epoch:  # safe checkpoint
            temp_loss = train_loss
            temp_net_state_dict = copy.deepcopy(net.state_dict())
            temp_optimizer_state_dict = copy.deepcopy(optimizer_net.state_dict())
            checkpoint_loss = train_loss

        if epoch > count_epoch:
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

        if end_flag > 5:
            break

        if (epoch + 1) % count_epoch == 0:
            result = denoise_image.squeeze()
            common_utils.plot_single_tensor_image(result)


    return train_loss, result

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

def denoise_DIP(SIM_SR):
    param_grid = {
        'learning_rate': [0.01],
        'batch_size': [1],
        # 'weight_decay': [1e-5],
        'weight_decay': [0],
        'Dropout_ratio': [1]
    }

    SIM_SR = SIM_SR / SIM_SR.max()

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    criterion = nn.MSELoss()

    denoise_net = UNet()

    save_file_directory, net_type, data_input_mode, LR_highway_type, num_epochs, image_size = load_hyper_params()
    image_size = SIM_SR.size()[0]
    experimental_params = funcs.SinusoidalPattern(probability=1, image_size=image_size)
    start_time = time.time()
    pertubation = common_utils.get_noise( 1, 'noise', image_size, var=0.1)


    train_loss, denoise_SIM_SR = train(denoise_net, SIM_SR, pertubation, criterion, num_epochs,
                                device, experimental_params, lr, weight_decay)
    end_time = time.time()
    denoise_SIM_SR_reshape = denoise_SIM_SR.reshape([1, 1, denoise_SIM_SR.size()[0], denoise_SIM_SR.size()[1]])

    SIM_SR_enhancement = SIM_SR * 2 - denoise_SIM_SR_reshape.detach().cpu()
    common_utils.save_image_tensor2pillow(abs(SIM_SR_enhancement), save_file_directory)
    # SIMnet.to('cpu')
    # torch.save(SIMnet.state_dict(), file_directory + '/SIMnet.pkl')
    print(
        'avg train rmse: %f, learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
        % (train_loss, lr, batch_size, weight_decay, Dropout_ratio, end_time - start_time))

    return denoise_SIM_SR


def load_hyper_params():
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
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

    return save_file_directory,net_type,data_input_mode,LR_highway_type,num_epochs,image_size


if __name__ == '__main__':
    noise_image_directoty = 'C:/Users/zenghui/Desktop/BAPEC/2021-10-21 22-45-24_SR_image.png'
    noise_image = cv2.imread(noise_image_directoty, -1) / 1.0

    noise_image_tensor = torch.from_numpy(noise_image).float()
    if len(noise_image_tensor.size()) == 3:
        noise_image_tensor_gray = noise_image_tensor[:,:,0]
    else:
        noise_image_tensor_gray = noise_image_tensor
    denoise_image = denoise_DIP(noise_image_tensor_gray)
