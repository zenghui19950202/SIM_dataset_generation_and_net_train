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
import copy
from torch.utils.data import DataLoader
import os
from utils.SpeckleSIMDataLoad import SIM_pattern_load
from utils.SpeckleSIMDataLoad import SIM_data_load
from simulation_data_generation import fuctions_for_generate_pattern as funcs
from torchvision import transforms
from self_supervised_learning_sr import common_utils
from simulation_data_generation.fuctions_for_generate_pattern import save_image_tensor2pillow
import numpy as np


def tv_loss_calculate(x, beta=0.5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))

def minus_loss_calculate(x):
    minus_x = torch.mean(abs(x) - x)

    return minus_x

def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:3')
    else:
        device = torch.device('cpu')
    return device


def train(net, SIM_data_loader, SIM_pattern_loader, net_input, criterion, num_epochs, device, lr=None, weight_decay=1e-5):
    """Train and evaluate a model with CPU or GPU."""

    print('training on', device)

    net = net.to(device)
    temp_net_parameters = []
    loss_in_5_epoch = torch.rand(5)
    temp_net_parameters += [net.state_dict()]+[net.state_dict()]+[net.state_dict()]+[net.state_dict()]+[net.state_dict()]

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-5, eps=1e-08)

    temp = funcs.SinusoidalPattern(probability=1)
    OTF = temp.OTF
    psf = temp.psf_form(OTF)
    psf_conv = funcs.psf_conv_generator(psf)

    noise = net_input.detach().clone()
    reg_noise_std = 0.03
    min_loss = 1e5
    image_size = [net_input.size()[2],net_input.size()[3]]
    best_SR = torch.zeros(image_size, dtype=torch.float32, device=device)
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        n, start = 0, time.time()
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        net_input_noise = net_input + (noise.normal_() * reg_noise_std)
        net_input_noise = net_input_noise.to(device)
        for SIM_data, SIM_pattern in zip(SIM_data_loader,SIM_pattern_loader):
            optimizer.zero_grad()
            SIM_raw_data = SIM_data[0]
            HR_LR = SIM_data[1]
            SIM_raw_data = SIM_raw_data.to(device)
            SIM_pattern = SIM_pattern.to(device)
            HR_LR = HR_LR.to(device)
            # temp = net(SIM_raw_data)
            SR_image = net(net_input_noise)
            SR_image = abs(SR_image)

            # Relu = nn.ReLU()
            # SR_image = Relu(SR_image)

            tv_loss = 1e-7* tv_loss_calculate(SR_image)
            # minus_loss = minus_loss_calculate(SR_image)
            SR_image = SR_image.squeeze()
            SIM_raw_data_estimated = positive_propagate(SR_image, SIM_pattern,psf_conv,device)
            # loss = criterion(SIM_raw_data[:,0:-1,:,:], SIM_raw_data_estimated)
            mse_loss = criterion(SIM_raw_data[:, 0:-1, :, :], SIM_raw_data_estimated[:, :, :,:])
            # LR = HR_LR[:,:,:,1]
            # LR_estimated = positive_propagate(SR_image, 1, psf_conv,device)
            # LR_mse_loss = criterion(LR_estimated,LR)
            # loss = tv_loss + minus_loss + mse_loss
            # loss = mse_loss
            loss = mse_loss + tv_loss
            # loss = minus_loss + mse_loss
            # loss = tv_loss + mse_loss + LR_mse_loss
            # loss =  mse_loss + LR_mse_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_l_sum += loss.float()
                n += 1
        train_loss = train_l_sum / n
        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))
        # print('epoch: %d/%d, train_loss: %f,lr: %s, SR_image_max, %f, SR_image_min, %f,, SR_image_mean, %f' % (epoch + 1, num_epochs, train_loss,optimizer.param_groups[0]['lr'],SR_image.max(),SR_image.min(),SR_image.mean()))
        # scheduler.step(train_loss)


        if epoch == 999: # safe checkpoint
            temp_loss = train_loss
            temp_net_state_dict = copy.deepcopy(net.state_dict())
            temp_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            checkpoint_loss = train_loss

        if epoch > 1000:
            delta_loss = train_loss - temp_loss
            temp_loss = train_loss
            print('delta_loss_max:%f, loss_min: %f'%(3 * delta_loss.max(),min(train_loss,temp_loss)))
            if 3 * delta_loss >  min(train_loss,temp_loss):
                net.load_state_dict(temp_net_state_dict)
                optimizer.load_state_dict(temp_optimizer_state_dict)
                temp_loss = checkpoint_loss
                print('revert:True')
            elif epoch % 50 == 0:
                    temp_net_state_dict = copy.deepcopy(net.state_dict())
                    temp_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
                    checkpoint_loss = train_loss

        if min_loss > train_loss:
            min_loss = train_loss
            best_SR = SR_image
        if (epoch+1) % 1000 == 0:
            # out_HR_np = common_utils.torch_to_np(SIM_raw_data_estimated)
            # out_HR_np = np.clip(out_HR_np, 0, 1)
            # # out_HR_np = out_HR_np/out_HR_np.max()
            # common_utils.plot_image_grid([out_HR_np[0, :, :].reshape(1, out_HR_np.shape[1], -1), out_HR_np[1, :, :].reshape(1, out_HR_np.shape[1], -1),
            #                               out_HR_np[2,:,:].reshape(1,out_HR_np.shape[1],-1)], factor=13, nrow=3)

            out_SR_np = SR_image.detach().cpu().numpy()
            out_SR_np = out_SR_np.reshape(1, out_SR_np.shape[0], -1)
            # out_SR_np = np.abs(out_SR_np)
            out_SR_np = out_SR_np/out_SR_np.max()
            common_utils.plot_image_grid([out_SR_np],factor=13, nrow=1)


    return train_loss, best_SR


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
        X, y = X.to(device), y.to(device)
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
        num_of_channels = pattern.size()[1]
        if pattern.mean() <1e-20:
            pattern_nomalized = pattern / (pattern.mean() + 1e-19)
        else:
            pattern_nomalized = pattern / pattern.mean()
        if pattern_gt.mean()<1e-20:
            pattern_gt_nomalized = pattern_gt / (pattern_gt.mean() + 1e-19)
        else:
            pattern_gt_nomalized = pattern_gt / pattern_gt.mean()
        loss = torch.mean(torch.pow((pattern_nomalized - pattern_gt_nomalized), 2))
        return loss * num_of_channels


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

    param_grid = {
        'learning_rate': [0.001],
        'batch_size': [1],
        'weight_decay': [1e-5],
        'Dropout_ratio': [1]
    }

    SIM_data = SIM_data_load(train_directory_file, normalize = False,data_mode = data_generate_mode)
    SIM_pattern = SIM_pattern_load(train_directory_file,normalize = False)
    SIM_data_dataloader = DataLoader(SIM_data,batch_size=1)
    SIM_pattern_dataloader = DataLoader(SIM_pattern,batch_size=1)

    random.seed(70)  # 设置随机种子
    # min_loss = 1e5
    num_epochs = 10000

    # directory_path = os.getcwd()
    # file_directory = directory_path + '/' + 'random_hyparameters' + time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())
    # if not os.path.exists(file_directory):
    #     os.makedirs(file_directory)
    # f_hyparameters = open(file_directory + "/hyperparams.txt", 'w')

    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    # criterion = nn.MSELoss()
    criterion = MSE_loss()
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

    net_input = common_utils.get_noise(3, 'noise', (image_size, image_size))
    net_input = net_input.to(device).detach()

    # net_input = SIM_data[0][1][:,:,0].squeeze()
    # net_input = torch.stack([net_input,net_input],0)
    # net_input = net_input.view(1,2,256,256)
    # net_input.requires_grad = True
    train_loss,best_SR = train(SIMnet, SIM_data_dataloader, SIM_pattern_dataloader,net_input,criterion, num_epochs, device, lr,
                                   weight_decay)
    best_SR = best_SR.reshape([1,image_size,image_size])
    save_image_tensor2pillow(best_SR,save_file_directory)
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
