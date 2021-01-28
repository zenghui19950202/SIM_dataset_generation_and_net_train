#!/usr/bin/env python
# -*- coding: utf-8 -*-
#target: To analyze the SIM wide-field images of different directions,
# author：zenghui time:2020/12/23

import torch
from torchvision import transforms
from utils.SpeckleSIMDataLoad import SIM_data_load
from torch.utils.data import DataLoader
# from utils import common_utils
from utils import load_configuration_parameters
from utils import common_utils
import os
from simulation_data_generation import fuctions_for_generate_pattern as funcs
import torch.fft as fft

def save_wide_field_image(input_tensor, file_name,direction ):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor/input_tensor.max()
    input_PIL = transforms.ToPILImage()(input_tensor.float())

    if not os.path.exists(file_name):
        try:
            os.makedirs(file_name)
        except IOError:
            print("Insufficient rights to read or write output directory (%s)"
                  % file_name)

    save_path = os.path.join(file_name, 'wide_field_direction'+ str(direction) + '.png')
    input_PIL.save(save_path)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def torch_2d_ifftshift(image):
    image = image.squeeze()
    if image.dim() == 2:
        shift = [-(image.shape[1 - ax] // 2) for ax in range(image.dim())]
        image_shift = torch.roll(image, shift, [1, 0])
    else:
        raise Exception('The dim of image must be 2')
    return image_shift

def torch_2d_fftshift(image):
    image = image.squeeze()
    if image.dim() == 2:
        shift = [image.shape[ax] // 2 for ax in range(image.dim())]
        image_shift = torch.roll(image, shift, [0, 1])
    else:
        raise Exception('The dim of image must be 2')
    return image_shift

if __name__ == '__main__':
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    data_generate_mode = train_net_parameters['data_generate_mode']
    net_type = train_net_parameters['net_type']
    data_input_mode = train_net_parameters['data_input_mode']
    LR_highway_type = train_net_parameters['LR_highway_type']
    data_num = train_net_parameters['data_num']

    train_directory_file = train_net_parameters['train_directory_file']
    save_file_directory = train_net_parameters['save_file_directory']
    SIM_train_dataset = SIM_data_load(train_directory_file,data_mode = data_generate_mode,normalize = True)

    # criterion = criterion = nn.MSELoss()
    SIM_valid_dataloader = DataLoader(SIM_train_dataset, batch_size=1, shuffle=True)
    # valid_loss = evaluate_valid_loss(SIM_valid_dataloader, criterion, SIMnet, device=torch.device('cpu'))
    # print('valid_loss:%f',valid_loss)
    a, b = SIM_train_dataset[0]
    wide_field_direction1 = a[0,:,:] + a[1,:,:] + a[2,:,:]
    wide_field_direction2 = a[3, :, :] + a[4, :, :] + a[5, :, :]
    wide_field_direction3 = a[6, :, :] + a[7, :, :] + a[8, :, :]

    mix_3_direction = a[0,:,:] + a[3,:,:] + a[6,:,:]


    image_size = [a.size()[1],a.size()[2]]
    temp = funcs.SinusoidalPattern(probability=1,image_size = image_size[0])
    OTF = temp.OTF
    wide_field_sum = a.sum(dim=0)

    deconv_fft = torch_2d_fftshift(fft.fftn(wide_field_sum)) * (OTF / (OTF * OTF + 0.04))
    deconv = fft.ifftn(torch_2d_ifftshift(deconv_fft))
    common_utils.plot_single_tensor_image(deconv)
    deconv= abs(deconv)

    # common_utils.plot_single_tensor_image(deconv)

    output_dir = os.path.dirname(train_directory_file)
    save_wide_field_image(wide_field_direction1.unsqueeze(0).unsqueeze(0),save_file_directory,1)
    save_wide_field_image(wide_field_direction2.unsqueeze(0).unsqueeze(0),save_file_directory,2)
    save_wide_field_image(wide_field_direction3.unsqueeze(0).unsqueeze(0),save_file_directory,3)
    save_wide_field_image(deconv.unsqueeze(0).unsqueeze(0), save_file_directory, 'deconv')
    # common_utils.plot_single_tensor_image(wide_field_direction1)
    # common_utils.plot_single_tensor_image(wide_field_direction2)
    # common_utils.plot_single_tensor_image(wide_field_direction3)


