# author: zenghui
# -*- coding: utf-8 -*-
# To solve the polarization wide field image using three phases method
# author：zenghui time:2021/4/13

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
import numpy.fft as fft

def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')
    return device

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

    input_id = 1
    data_id = 0
    for SIM_data, SIM_pattern in zip(SIM_data_dataloader, SIM_pattern_dataloader):
        # SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1
    SIM_raw_data = SIM_data[0].squeeze()

    wide_field_1 = torch.mean(SIM_raw_data[0:3,:,:],dim=0)
    wide_field_2 = torch.mean(SIM_raw_data[3:6,:,:],dim=0)
    wide_field_3 = torch.mean(SIM_raw_data[6:9,:,:],dim=0)
    # common_utils.plot_single_tensor_image(wide_field_1)
    # common_utils.plot_single_tensor_image(wide_field_2)
    # common_utils.plot_single_tensor_image(wide_field_3)

    SIM_data_three_direction = torch.stack([SIM_data[0][:, 0, :, :], SIM_data[0][:, 3, :, :], SIM_data[0][:, 6, :, :]],dim=1)
    _, estimated_pattern_parameters,_ = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
        SIM_data_three_direction)
    # print(estimated_pattern_parameters)
    theta = torch.atan(estimated_pattern_parameters[:, 1] / estimated_pattern_parameters[:, 0])
    wide_field_np_1, wide_field_np_2, wide_field_np_3= wide_field_1.numpy(), wide_field_2.numpy(), wide_field_3.numpy()
    fft_wide_field_np_1, fft_wide_field_np_2, fft_wide_field_np_3 = fft.fftshift(fft.fft2(wide_field_np_1)), fft.fftshift(fft.fft2(wide_field_np_2)), fft.fftshift(fft.fft2(wide_field_np_3))
    M_matrix = np.array([[1,1/2*np.exp(1j*2*theta[0]),1/2*np.exp(-2*1j*theta[0])],[1,1/2*np.exp(2*1j*theta[1]),1/2*np.exp(-2*1j*theta[1])],[1,1/2*np.exp(2*1j*theta[2]),1/2*np.exp(-2*1j*theta[2])]])
    M_matrix_inv = np.linalg.inv(M_matrix)
    fft_wide_field_np = M_matrix_inv[0,0] * fft_wide_field_np_1 + M_matrix_inv[0,1] * fft_wide_field_np_2 + M_matrix_inv[0,2] * fft_wide_field_np_3
    wide_field_np = abs( fft.ifft2(fft.ifftshift(fft_wide_field_np)) )
    wide_field = torch.from_numpy(wide_field_np)
    # polarization_raio = torch.zeros_like(SIM_data_three_direction.squeeze())
    polarization_raio = torch.stack([wide_field_1/wide_field, wide_field_2/wide_field, wide_field_3/wide_field],dim=0)

    plt.imshow(wide_field_np,cmap ='gray')
    plt.show()
    a = wide_field_np_1 - wide_field_np
    a = a/abs(a).max()
    phi = np.arccos(a)
    phi += 2 * theta[0].numpy()

    a = wide_field_np_2 - wide_field_np
    a = a/abs(a).max()
    phi = np.arccos(a)
    phi += 2 * theta[1].numpy()

    a = wide_field_np_3 - wide_field_np
    a = a/abs(a).max()
    phi = np.arccos(a)
    phi += 2 * theta[2].numpy()

    plt.imshow(np.cos(phi), cmap='gray')
    plt.show()

