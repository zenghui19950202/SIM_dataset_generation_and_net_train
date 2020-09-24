#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/9
from utils import *
from models import *
import torch
import torch.optim as optim
import torch.nn as nn
import random
import time

from torch.utils.data import DataLoader
import os
from utils.SpeckleSIMDataLoad import SIM_pattern_load
from utils.SpeckleSIMDataLoad import SIM_data_load
from simulation_data_generation import fuctions_for_generate_pattern as funcs
from torchvision import transforms
from self_supervised_learning_sr import common_utils
import numpy as np

def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def train(net, SIM_data_loader, SIM_pattern_loader, net_input, criterion, num_epochs, device, lr=None, weight_decay=1e-5):
    """Train and evaluate a model with CPU or GPU."""

    print('training on', device)

    net = net.to(device)

    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    # 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
    # 因此可以通过名字判断属性，这个和tensorflow不同，tensorflow是可以用户自己定义名字的，当然也会系统自己定义。
    optimizer = optim.Adam([
        {'params': weight_p, 'weight_decay': weight_decay},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=lr)

    # optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)

    temp = funcs.SinusoidalPattern(probability=1)
    OTF = temp.OTF
    psf = temp.psf_form(OTF)
    psf_conv = funcs.psf_conv_generator(psf)

    noise = net_input.detach().clone()
    reg_noise_std = 0.03
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        n, start = 0, time.time()
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        net_input_noise = net_input + (noise.normal_() * reg_noise_std)
        for SIM_data, SIM_pattern in zip(SIM_data_loader,SIM_pattern_loader):
            optimizer.zero_grad()
            SIM_raw_data = SIM_data[0]
            SIM_raw_data = SIM_raw_data.to(device)
            SIM_pattern = SIM_pattern.to(device)
            # temp = net(SIM_raw_data)
            # Relu = nn.ReLU()
            # SR_image =  Relu(temp)
            SR_image = net(net_input_noise)
            SR_image = SR_image.squeeze()
            SIM_raw_data_estimated = positive_propagate(SR_image, SIM_pattern,psf_conv,device)
            loss = criterion(SIM_raw_data[:,0:-1,:,:], SIM_raw_data_estimated)
            # loss = criterion(SIM_raw_data[:, -1, :, :], SR_image)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_l_sum += loss.float()
                n += 1
        train_loss = train_l_sum / n
        print('epoch: %d/%d, train_loss: %f ' % (epoch + 1, num_epochs, train_loss))
        # scheduler.step(train_loss)

        if (epoch+1) % 100 == 0:
            out_HR_np = common_utils.torch_to_np(SIM_raw_data_estimated)
            out_HR_np = np.clip(out_HR_np, 0, 1)
            common_utils.plot_image_grid([out_HR_np[0, :, :].reshape(1, out_HR_np.shape[1], -1), out_HR_np[1, :, :].reshape(1, out_HR_np.shape[1], -1),
                                          out_HR_np[2,:,:].reshape(1,out_HR_np.shape[1],-1)], factor=13, nrow=3)

    SR_image = net(SIM_raw_data)
    SR_image = SR_image.squeeze()
    # SR_image = Relu(SR_image)
    SIM_raw_data_estimated = positive_propagate(SR_image, SIM_pattern, psf_conv, device)
    SIM_raw_data_estimated = SIM_raw_data_estimated.cpu().squeeze()
    SR_image = SR_image.cpu().squeeze()
    SR_image_PIL = transforms.ToPILImage()(SR_image)

    SIM_raw_data_estimated_PIL = transforms.ToPILImage()(SIM_raw_data_estimated[4,:,:])
    # SR_image_PIL.show()
    SIM_raw_data_estimated_PIL.show()

    return train_loss


def positive_propagate(SR_image,SIM_pattern,psf_conv,device):
    SIM_raw_data_estimated = psf_conv(SR_image*SIM_pattern,device)
    return SIM_raw_data_estimated

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def evaluate_valid_loss(data_iter, criterion, net, device=torch.device('cpu')):
    net.eval()
    """Evaluate accuracy of a model on the given data set."""
    loss_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        X, y = X.cuda(), y.cuda()
        with torch.no_grad():
            y = y.float()
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            loss_sum += criterion(y_hat, y)
            n += 1
    return loss_sum.item() / n


class SR_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, SR_image, HR_LR_image, loss_ratio=0.8):
        if len(HR_LR_image.shape) == 4:
            loss = (1 - loss_ratio) * torch.mean(
                torch.pow((SR_image - HR_LR_image[:, :, :, 0]), 2)) + loss_ratio * torch.mean(
                torch.pow((SR_image - HR_LR_image[:, :, :, 0] + HR_LR_image[:, :, :, 1]), 2))
        elif len(HR_LR_image.shape) == 3:
            loss = (1 - loss_ratio) * torch.mean(
                torch.pow((SR_image - HR_LR_image[:, :, 0]), 2)) + loss_ratio * torch.mean(
                torch.pow((SR_image - HR_LR_image[:, :, 0] + HR_LR_image[:, :, 1]), 2))
        return loss


class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pattern, pattern_gt):
        loss = torch.mean(torch.pow((pattern - pattern_gt), 2))
        return loss


if __name__ == '__main__':

    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    train_directory_file = train_net_parameters['train_directory_file']
    valid_directory_file = train_net_parameters['valid_directory_file']
    data_generate_mode = train_net_parameters['data_generate_mode']
    net_type = train_net_parameters['net_type']
    data_input_mode = train_net_parameters['data_input_mode']
    LR_highway_type = train_net_parameters['LR_highway_type']
    MAX_EVALS = train_net_parameters['MAX_EVALS']
    num_epochs = train_net_parameters['num_epochs']
    data_num = train_net_parameters['data_num']
    image_size = train_net_parameters['image_size']

    param_grid = {
        'learning_rate': [0.001],
        'batch_size': [1],
        'weight_decay': [1e-5],
        'Dropout_ratio': [1]
    }

    SIM_data = SIM_data_load(train_directory_file, normalize = False)
    SIM_pattern = SIM_pattern_load(train_directory_file)
    SIM_data_dataloader = DataLoader(SIM_data,batch_size=1)
    SIM_pattern_dataloader = DataLoader(SIM_pattern,batch_size=1)

    random.seed(70)  # 设置随机种子
    min_loss = 1e5
    num_epochs = 100

    directory_path = os.getcwd()
    file_directory = directory_path + '/' + 'random_hyparameters' + time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    f_hyparameters = open(file_directory + "/hyperparams.txt", 'w')

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    # criterion = nn.MSELoss()
    criterion = MSE_loss()
    num_raw_SIMdata, output_nc, num_downs = 31, 1, 5
    # SIMnet = Networks_Unet_GAN.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=False,input_mode = 'only_input_SIM_images', use_dropout=False)
    SIMnet = resnet_backbone_net._resnet('resnet34', resnet_backbone_net.BasicBlock, [1, 1, 1, 1], input_mode='only_input_SIM_images',
                                LR_highway=False, input_nc=num_raw_SIMdata, pretrained=False, progress=False, )
    # SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, 1, input_mode=data_input_mode, LR_highway=LR_highway_type)
    SIMnet.apply(init_weights)
    start_time = time.time()

    net_input = common_utils.get_noise(32, 'noise', (image_size, image_size)).type(torch.cuda.FloatTensor).detach()
    train_loss = train(SIMnet, SIM_data_dataloader, SIM_pattern_dataloader,net_input,criterion, num_epochs, device, lr,
                                   weight_decay)

    # SIMnet.to('cpu')
    end_time = time.time()
    torch.save(SIMnet.state_dict(), file_directory + '/SIMnet.pkl')
    print(
        'avg train rmse: %f, learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
        % (train_loss, lr, batch_size, weight_decay, Dropout_ratio, end_time - start_time))

    # a = SIM_train_dataset[0]
    # image = a[0]
    # image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    # print(SIMnet(image1))
