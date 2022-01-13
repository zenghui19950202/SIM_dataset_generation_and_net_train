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
from parameter_estimation.estimate_polarizaion import calculate_polarization_ratio
from Deep_image_prior import Unet_for_denoise

# import self_supervised_learning_sr.estimate_SIM_pattern
import numpy as np


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def train(net, SIM_raw_data, polarization_ratio, net_input, criterion, num_epochs, device, experimental_parameters, lr=None,
          weight_decay=1e-5):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net = net.to(device)
    if experimental_parameters.upsample:
        OTF = experimental_parameters.OTF_upsmaple
    else:
        OTF = experimental_parameters.OTF
    psf = experimental_parameters.psf_form(OTF)

    psf_conv = funcs.psf_conv_generator(psf,device)

    reg_noise_std = 0.03
    min_loss = 1e5
    image_size = [net_input.size()[2], net_input.size()[3]]
    best_SR = torch.zeros(image_size, dtype=torch.float32, device=device)
    end_flag = 0

    input_num = 6
    input_SIM_raw_data = common_utils.input_data_pick(SIM_raw_data,input_num)
    polarization_ratio = common_utils.input_data_pick(polarization_ratio, input_num)

    psf_radius = math.floor(psf.size()[0] / 2)

    # temp_input_SIM_pattern, estimated_pattern_parameters, estimated_SIM_pattern_without_m = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
    #     input_SIM_raw_data,experimental_parameters)

    temp_input_SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_TIRF_image(input_SIM_raw_data, SIM_raw_data, experimental_parameters)


    print(estimated_pattern_parameters)

    xx, yy, _, _ = experimental_parameters.GridGenerate( grid_mode='pixel')

    delta_pattern_params = torch.zeros_like(estimated_pattern_parameters)

    fusion_param = torch.tensor([1.0],device = device)

    net_parameters = common_utils.get_params('net', net,fusion_param, delta_pattern_params, downsampler=None,
                                                  weight_decay=weight_decay)
    params = []
    delta_pattern_params.requires_grad = True
    params += [{'params': delta_pattern_params, 'weight_decay': weight_decay}]
    optimizer_net = optim.Adam(net_parameters, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, mode='min', factor=0.9, patience=100,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-5, eps=1e-08)

    if experimental_parameters.upsample == True:
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        input_polarization_ratio = upsample(polarization_ratio).to(device)
        input_SIM_raw_data = upsample(input_SIM_raw_data).to(device)
        net_input = upsample(net_input).to(device)
    else:
        input_polarization_ratio = polarization_ratio.to(device)
        input_SIM_raw_data = input_SIM_raw_data.to(device)

    noise = net_input.detach().clone()

    mask = torch.zeros_like(input_SIM_raw_data, device=device)
    mask[:, :, psf_radius:-psf_radius, psf_radius:-psf_radius] = 1
    mask[:]=1

    temp_input_SIM_pattern = temp_input_SIM_pattern.to(device)
    net = net.to(device)

    count_epoch = 0
    switch_flag = 1
    min_loss = 1e5
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        net_input_noise = net_input + (noise.normal_() * reg_noise_std)
        net_input_noise = net_input_noise.to(device)

        optimizer_net.zero_grad()

        SR_image = net(net_input_noise)
        SR_image = abs(SR_image)
        tv_loss = 1e-7 * loss_functions.tv_loss_calculate(SR_image)
        tv_loss=0
        loss = tv_loss
        # SIM_raw_data_estimated = forward_model.positive_propagate(SR_image * input_polarization_ratio, temp_input_SIM_pattern.detach(),psf_conv)
        SIM_raw_data_estimated = forward_model.positive_propagate(SR_image,temp_input_SIM_pattern.detach(), psf_conv)
        mse_loss = criterion(SIM_raw_data_estimated, input_SIM_raw_data, mask,normalize = False)
        loss+=mse_loss

        loss.backward()
        optimizer_net.step()

        with torch.no_grad():
            train_loss = loss.float()

        output_count = 500

        print('epoch: %d/%d, train_loss: %f  MSE_loss:%f , tv_loss: %f' % (epoch + 1, num_epochs, train_loss, mse_loss,tv_loss ))
        # SIM_pattern = estimate_SIM_pattern.fine_adjust_SIM_pattern(input_SIM_raw_data,estimated_pattern_parameters,delta_pattern_params,xx,yy)
        # print(delta_pattern_params)
        if epoch == output_count-1:  # safe checkpoint
            temp_loss = train_loss
            temp_net_state_dict = copy.deepcopy(net.state_dict())
            temp_optimizer_state_dict = copy.deepcopy(optimizer_net.state_dict())
            checkpoint_loss = train_loss

        if epoch > output_count:
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

        # if epoch > 2000:
        #     wide_field_estimated = forward_model.positive_propagate(SR_image, 1, psf_conv)
        #
        # if min_loss > train_loss:
        #     min_loss = train_loss
        #
        #     # result = processing_utils.notch_filter(SR_image_high_freq_filtered, estimated_pattern_parameters)
        #     # best_SR =SR_image_high_freq_filtered
        #     best_SR = result[psf_radius:-psf_radius, psf_radius:-psf_radius]


        if end_flag > 5:
            break

        if (epoch + 1) % output_count == 0:
            result = processing_utils.notch_filter_for_all_vulnerable_point(abs(SR_image),estimated_pattern_parameters,
                                                                            experimental_parameters).squeeze()
            # result = SR_image_high_freq_filtered[psf_radius:-psf_radius, psf_radius:-psf_radius]
            common_utils.plot_single_tensor_image(result)


    return train_loss, result

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

def train_net():
    param_grid = {
        'learning_rate': [0.001],
        'batch_size': [1],
        # 'weight_decay': [1e-5],
        'weight_decay': [0],
        'Dropout_ratio': [1]
    }

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    criterion = loss_functions.MSE_loss()
    num_raw_SIMdata, output_nc, num_downs = 3, 1, 5

    SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, output_nc, input_mode='input_all_images', LR_highway=False)
    SIMnet = Unet_for_denoise.UNet()


    save_file_directory, net_type, data_input_mode, LR_highway_type, num_epochs, image_size = load_hyper_params()



    input_SIM_data,input_polarization_ratio,net_input,experimental_params = initialize_data(num_raw_SIMdata)

    net_input = common_utils.get_noise(1, 'noise', image_size, var=0.1)

    start_time = time.time()
    train_loss, best_SR = train(SIMnet, input_SIM_data, input_polarization_ratio, net_input, criterion, num_epochs,
                                device, experimental_params, lr, weight_decay)
    end_time = time.time()
    best_SR = best_SR.reshape([1, 1, best_SR.size()[0], best_SR.size()[1]])
    common_utils.save_image_tensor2pillow(best_SR, save_file_directory)
    # SIMnet.to('cpu')
    # torch.save(SIMnet.state_dict(), file_directory + '/SIMnet.pkl')
    print(
        'avg train rmse: %f, learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
        % (train_loss, lr, batch_size, weight_decay, Dropout_ratio, end_time - start_time))

    # a = SIM_train_dataset[0]
    # image = a[0]
    # image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    # print(SIMnet(image1))

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

def initialize_data(num_raw_SIMdata = 2):
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    train_directory_file = train_net_parameters['train_directory_file']

    SIM_data = SpeckleSIMDataLoad.SIM_data_load(train_directory_file, normalize=True, data_mode='only_raw_SIM_data')

    SIM_data_dataloader = DataLoader(SIM_data, batch_size=1)


    random.seed(60)  # 设置随机种子

    input_id = 1
    data_id = 0
    for input_SIM_data in SIM_data_dataloader:
        # SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1

    LR_HR = input_SIM_data[1]
    SIM_raw_data = input_SIM_data[0]



    image_size = torch.tensor([SIM_raw_data.size()[2], SIM_raw_data.size()[3]])
    experimental_params = funcs.SinusoidalPattern(probability=1, image_size=image_size[0])
    polarization_ratio = calculate_polarization_ratio(SIM_raw_data, experimental_params)
    if experimental_params.upsample == True:
        image_size *= 2

    net_input = common_utils.get_noise(num_raw_SIMdata+1, 'noise', image_size, var=0.1)

    return SIM_raw_data,polarization_ratio,net_input,experimental_params

if __name__ == '__main__':
    train_net()
