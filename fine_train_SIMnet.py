#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/7/1

import torch
import torch.optim as optim
import torch.nn as nn
# import numpy as np
# import random
import time
from utils.SpeckleSIMDataLoad import SIM_data_load
from torch.utils.data import DataLoader
# from Unet_NC2020 import UNet
import os
from tensorboardX import SummaryWriter
from models.Networks_Unet_GAN import UnetGenerator
from early_stopping.pytorchtools import EarlyStopping
from models import resnet_backbone_net as res_SIMnet
import math
from configparser import ConfigParser

# from visdom import Visdom


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def train(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None, weight_decay = 1e-4,logfile_directory=None):
    """Train and evaluate a model with CPU or GPU."""
    # vis = Visdom(env='model_1')
    # win = vis.line(X=np.array([0]), Y=np.array([0]), name="1")
    writer = SummaryWriter(logfile_directory)
    print('training on', device)
    net.to(device)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10,
    verbose=False, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=False)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        n, start = 0, time.time()
        ram_time = 0
        GPU_time = 0
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        for X, y in train_iter:
            optimizer.zero_grad()
            # y = y[:,:,:,0]
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            loss,_ = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.float()
                train_l_sum += loss.float()
                n += 1
        train_loss = train_l_sum / n
        valid_loss, PSNR = evaluate_valid_loss(test_iter,criterion, net, device)
        scheduler.step(valid_loss)

        print('epoch %d, train_loss %f,valid_loss %f, PSNR: %f, time %.1f sec' \
              % (epoch + 1, train_loss,valid_loss,PSNR, time.time() - start))
        early_stopping(valid_loss,net,epoch)
        writer.add_scalars('scalar/loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch + 1)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # vis.updateTrace(X=epoch + 1, Y=train_l_sum / n, win=win, name="1")  # TODO: visdom 可视化
        # vis.line(X= epoch_tensor, Y= torch.tensor([min(3,train_l_sum / n)]), win=win, update='append')
    writer.close
    return train_loss, valid_loss

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def evaluate_valid_loss(data_iter,criterion, net, device=torch.device('cpu')):
    net.eval()
    """Evaluate accuracy of a model on the given data set."""
    loss_sum, PSNR_sum, n = torch.tensor([0], dtype=torch.float32, device=device),torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        # y = y[:,:,:,0]
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.float()
            y_hat=net(X)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            # loss_sum += criterion(y_hat, y)
            loss , PSNR = criterion(y_hat, y)
            loss_sum += loss
            PSNR_sum += PSNR
            n += 1
    return loss_sum.item() / n, PSNR_sum.item()/n

class SR_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, SR_image, HR_LR_image,loss_ratio = 0.8):
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
    def forward(self, SR_image, HR_LR_image):
        if len(HR_LR_image.shape) == 4:
            HR = HR_LR_image[:, :, :, 0]
            loss = torch.mean(
                torch.pow((SR_image -HR), 2))
        elif len(HR_LR_image.shape) == 3:
            HR = HR_LR_image[:, :, 0]
            loss = torch.mean(
                torch.pow((SR_image - HR), 2))

        PSNR = 20 * math.log10(HR.max() / math.sqrt(loss))
        return loss,PSNR

if __name__ == '__main__':
    config = ConfigParser()
    config.read('configuration.ini')
    SourceFileDirectory = config.get('image_file', 'SourceFileDirectory')

    train_directory_file = SourceFileDirectory + '/SIMdata_SR_train.txt'
    valid_directory_file = SourceFileDirectory + '/SIMdata_SR_valid.txt'

    data_generate_mode = config.get('data', 'data_generate_mode')
    data_input_mode = config.get('data', 'data_input_mode')
    net_type = config.get('net', 'net_type')
    LR_highway_type = config.get('LR_highway', 'LR_highway_type')

    SIM_train_dataset = SIM_data_load(train_directory_file, data_mode = data_generate_mode)
    SIM_valid_dataset = SIM_data_load(valid_directory_file, data_mode = data_generate_mode)

    num_epochs =2

    random_params = {
        'learning_rate':  0.000167,
        'batch_size': 32,
        'Dropout_ratio': 1,
        'weight_decay': 1e-5
    }
    lr = random_params['learning_rate']
    batch_size = random_params['batch_size']
    weight_decay = random_params['weight_decay']
    Dropout_ratio = random_params['Dropout_ratio']

    device = try_gpu()
    # criterion = nn.MSELoss()
    # criterion = SR_loss()
    criterion = MSE_loss()

    # SIM_train_dataloader = DataLoader(SIM_train_dataset,num_workers=8,pin_memory=True, batch_size=batch_size, shuffle=True)
    # SIM_valid_dataloader = DataLoader(SIM_valid_dataset,num_workers=8,pin_memory=True, batch_size=batch_size, shuffle=True)
    SIM_train_dataloader = DataLoader(SIM_train_dataset, batch_size=batch_size, shuffle=True)
    SIM_valid_dataloader = DataLoader(SIM_valid_dataset, batch_size=batch_size, shuffle=True)

    directory_path = os.path.abspath(os.path.dirname(os.getcwd()))
    file_directory = directory_path + '/train_result/' + net_type + '_' + data_generate_mode + '_' + data_input_mode + '_' + LR_highway_type + '_' + time.strftime(
        "%Y_%m_%d %H_%M_%S", time.localtime())

    # directory_path = os.getcwd()
    # file_name = 'temp'
    # file_directory = directory_path + '/' + file_name
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    logfile_directory = file_directory+'/log_file'
    num_raw_SIMdata, output_nc, num_downs = 9, 1, 5
    if net_type == 'Unet':
        SIMnet = UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=LR_highway_type,
                               input_mode=data_input_mode, use_dropout=False)
    elif net_type == 'resnet':
        SIMnet = res_SIMnet._resnet('resnet34', res_SIMnet.BasicBlock, [1, 1, 1, 1],
                                    input_mode=data_input_mode, LR_highway=LR_highway_type,
                                    input_nc=num_raw_SIMdata,
                                    pretrained=False, progress=False, )
    SIMnet.apply(init_weights)
    start_time = time.time()
    train_loss, valid_loss = train(SIMnet, SIM_train_dataloader, SIM_valid_dataloader, criterion, num_epochs, batch_size, device, lr,weight_decay,logfile_directory)
    # SIMnet.to('cpu')
    end_time = time.time()

    torch.save(SIMnet.state_dict(), file_directory+ '/SIMnet.pkl')

    file_name = ('lr_' + str(lr) + 'num_epochs_' + str(num_epochs) + 'batch_size_' + str(
        batch_size) + 'weight_decay_' + str(weight_decay)+str(valid_loss))
    file_directory_new = directory_path + '/' + file_name

    os.rename(file_directory,file_directory_new)


    # torch.save(SIMnet.state_dict(), file_directory+ '/SIMnet.pkl')

    # a = SIM_train_dataset[0]
    # image = a[0]
    # image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    # print(SIMnet(image1))


