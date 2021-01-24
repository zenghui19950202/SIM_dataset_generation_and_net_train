#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2021/1/14

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/9
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

    OTF_reconstruction = temp.OTF_form(fc_ratio=1 + temp.pattern_frequency_ratio)
    psf_reconstruction = temp.psf_form(OTF_reconstruction)
    psf_reconstruction_conv = funcs.psf_conv_generator(psf_reconstruction,device)

    noise = net_input.detach().clone()
    reg_noise_std = 0.03
    min_loss = 1e5
    image_size = [net_input.size()[2], net_input.size()[3]]
    best_SR = torch.zeros(image_size, dtype=torch.float32, device=device)
    end_flag = 0

    input_id = 1
    data_id = 0
    for SIM_data, SIM_pattern in zip(SIM_data_dataloader, SIM_pattern_loader):
        SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1

    input_SIM_raw_data = common_utils.pick_input_data(SIM_raw_data,[0,3,6])
    input_SIM_pattern = common_utils.pick_input_data(SIM_pattern,[0,3,6])
    # input_SIM_raw_data_normalized = processing_utils.pre_processing(input_SIM_raw_data)

    psf_radius = math.floor(psf.size()[0] / 2)

    mask = torch.zeros_like(input_SIM_raw_data, device=device)
    mask[:, :, psf_radius:-psf_radius, psf_radius:-psf_radius] = 1

    temp_input_SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels(
        input_SIM_raw_data)
    print(estimated_pattern_parameters)
    experimental_parameters = SinusoidalPattern(probability=1)
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size[0], grid_mode='pixel')

    delta_pattern_params = torch.zeros_like(estimated_pattern_parameters)

    fusion_param = torch.tensor([1.0],device = device)

    net_parameters = common_utils.get_params('net', net,fusion_param, delta_pattern_params, downsampler=None,
                                                  weight_decay=weight_decay)
    params = []
    delta_pattern_params.requires_grad = True
    params += [{'params': delta_pattern_params, 'weight_decay': weight_decay}]
    optimizer_pattern_params = optim.Adam(params, lr=0.01)

    optimizer_net = optim.Adam(net_parameters, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, mode='min', factor=0.9, patience=100,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-5, eps=1e-08)
    input_SIM_raw_data = input_SIM_raw_data.to(device)
    input_SIM_pattern = input_SIM_pattern.to(device)
    temp_input_SIM_pattern = temp_input_SIM_pattern.to(device)
    temp_input_SIM_pattern = input_SIM_pattern

    count_epoch = 0
    switch_flag = 1
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        net_input_noise = net_input + (noise.normal_() * reg_noise_std)
        net_input_noise = net_input_noise.to(device)

        optimizer_net.zero_grad()
        optimizer_pattern_params.zero_grad()

        SR_image = net(net_input_noise)
        SR_image = abs(SR_image)
        SR_image_high_freq_filtered = torch.zeros_like(SR_image.squeeze())
        tv_loss = 1e-7 * loss_functions.tv_loss_calculate(SR_image)
        tv_loss=0
        loss = tv_loss
        for direction in range(3):
            SR_image_direction = SR_image[:,direction,:,:]
            SIM_raw_data_estimated = forward_model.positive_propagate(SR_image_direction, temp_input_SIM_pattern.detach(),
                                                                      psf_conv)
            SR_image_high_freq_filtered[direction,:,:] = forward_model.positive_propagate(SR_image_direction, 1,
                                                                           psf_reconstruction_conv)
            mse_loss = criterion(SIM_raw_data_estimated, input_SIM_raw_data, mask)
            loss+=mse_loss

        loss.backward()
        optimizer_net.step()


        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f  MSE_loss:%f , tv_loss: %f' % (epoch + 1, num_epochs, train_loss, mse_loss,tv_loss ))
        # SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern(input_SIM_raw_data,estimated_pattern_parameters,delta_pattern_params,xx,yy)
        # print(delta_pattern_params)
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

        if epoch > 2000:
            wide_field_estimated = forward_model.positive_propagate(SR_image, 1, psf_conv)

        if min_loss > train_loss:
            min_loss = train_loss
            result = torch.mean(SR_image_high_freq_filtered,0)
            # result = processing_utils.notch_filter(SR_image_high_freq_filtered, estimated_pattern_parameters)
            # best_SR =SR_image_high_freq_filtered
            best_SR = result[psf_radius:-psf_radius, psf_radius:-psf_radius]


        if end_flag > 5:
            break

        if (epoch + 1) % 1000 == 0:
            result = processing_utils.notch_filter_for_all_vulnerable_point(torch.mean(SR_image_high_freq_filtered,0),
                                                                            estimated_pattern_parameters).squeeze()[psf_radius:-psf_radius, psf_radius:-psf_radius]
            # result = SR_image_high_freq_filtered[psf_radius:-psf_radius, psf_radius:-psf_radius]
            common_utils.plot_single_tensor_image(torch.mean(SR_image_high_freq_filtered,0))


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
    num_epochs = 3000

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    # criterion = nn.MSELoss()
    criterion = loss_functions.MSE_loss()
    num_raw_SIMdata, output_nc, num_downs = 2, 3, 5
    # SIMnet = Unet_for_self_supervised.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,input_mode = 'only_input_SIM_images', use_dropout=False)
    # SIMnet = Networks_Unet_GAN.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,
    #                                                 input_mode='only_input_SIM_images', use_dropout=False)
    SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, output_nc, input_mode='input_all_images', LR_highway=False)
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

    # a = SIM_train_dataset[0]
    # image = a[0]
    # image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    # print(SIMnet(image1))
