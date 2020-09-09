#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/16

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import time
from SpeckleSIMDataLoad import SIM_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import resnet_backbone_net as res_SIMnet
# from Unet_NC2020 import UNet
import os
# from tensorboardX import SummaryWriter
from Networks_Unet_GAN import UnetGenerator
from early_stopping.pytorchtools import EarlyStopping
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
    # writer = SummaryWriter(logfile_directory)
    print('training on', device)
    # nn.DataParallel(net,device_ids=[0,1,2,3])
    # nn.DataParallel(net, device_ids=[0])
    net = net.cuda()
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4,
    verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


    epoch_tensor = torch.rand(1)

    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=False)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    with tqdm(total=num_epochs, desc="Executing train", unit=" epoch") as progress_bar:
        for epoch in range(num_epochs):
            net.train()  # Switch to training mode
            n, start = 0, time.time()
            train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
            train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
            for X, y in train_iter:
                optimizer.zero_grad()
                X, y = X.cuda(), y.cuda()
                y_hat = net(X)
                y_hat = y_hat.squeeze()
                y = y.squeeze()
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    y = y.float()
                    train_l_sum += loss.float()
                    n += y.shape[0]
            train_loss = train_l_sum / n
            valid_loss = evaluate_valid_loss(test_iter,criterion, net, device)
            # scheduler.step(valid_loss)
            # print('epoch %d, loss %.4f,valid_loss %.4f, time %.1f sec' \
            #       % (epoch + 1, train_l_sum / n,valid_loss, time.time() - start))
            early_stopping(valid_loss,net,epoch)
            # writer.add_scalars('scalar/loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch + 1)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            progress_bar.set_description("epoch" )
            progress_bar.update(1)
    # writer.close
    return train_loss, valid_loss

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def evaluate_valid_loss(data_iter,criterion, net, device=torch.device('cpu')):
    net.eval()
    """Evaluate accuracy of a model on the given data set."""
    loss_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        X, y = X.cuda(), y.cuda()
        with torch.no_grad():
            y = y.float()
            y_hat=net(X)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            loss_sum += criterion(y_hat, y)
            n += y.shape[0]
    return loss_sum.item() / n

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
            loss = torch.mean(
                torch.pow((SR_image - HR_LR_image[:, :, :, 0]), 2))
        elif len(HR_LR_image.shape) == 3:
            loss = torch.mean(
                torch.pow((SR_image - HR_LR_image[:, :, 0]), 2))
        return loss


if __name__ == '__main__':
    config = ConfigParser()
    config.read('configuration.ini')
    SourceFileDirectory = config.get('image_file', 'SourceFileDirectory')

    # train_directory_file = os.path.dirname(SourceFileDirectory)+'/train.txt'
    # valid_directory_file = os.path.dirname(SourceFileDirectory)+'/valid.txt'
    train_directory_file = SourceFileDirectory + '/train.txt'
    valid_directory_file = SourceFileDirectory + '/valid.txt'

    MAX_EVALS = config.getint('hyparameter', 'MAX_EVALS') # the number of hyparameters sets
    MAX_EVALS = 2

    # train_directory_file = 'D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/train.txt'
    # valid_directory_file = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/valid.txt"
    # train_directory_file = '/home/yyk/zenghui2020/multispot_SIM_dataset/train.txt'
    # valid_directory_file = "/home/yyk/zenghui2020/multispot_SIM_dataset/valid.txt"
    param_grid = {
        'learning_rate': list(np.logspace(-4, -2, base=10, num=10)),
        'batch_size': [8, 16, 32],
        'weight_decay': [1e-5],
        'Dropout_ratio':  [1]
    }

    SIM_train_dataset = SIM_data(train_directory_file,data_mode = 'SIM_and_sum_images')
    SIM_valid_dataset = SIM_data(valid_directory_file,data_mode = 'SIM_and_sum_images')

    random.seed(70)  # 设置随机种子
    min_loss=1e5
    num_epochs =2

    directory_path = os.getcwd()
    file_directory = directory_path + '/' + 'random_hyparameters'+ time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    f_hyparameters = open(file_directory + "/hyperparams.txt", 'w')
    for i in range(2):

        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        lr = random_params['learning_rate']
        batch_size = random_params['batch_size']
        weight_decay = random_params['weight_decay']
        Dropout_ratio = random_params['Dropout_ratio']

        device = try_gpu()
        # criterion = nn.MSELoss()
        criterion = MSE_loss()

        SIM_train_dataloader = DataLoader(SIM_train_dataset, batch_size=batch_size, shuffle=True)
        SIM_valid_dataloader = DataLoader(SIM_valid_dataset, batch_size=batch_size, shuffle=True)

        num_raw_SIMdata, output_nc, num_downs = 9, 1, 5
        SIMnet = UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway='add',input_mode = 'only_input_SIM_images', use_dropout=False)
        # SIMnet = res_SIMnet._resnet('resnet34', res_SIMnet.BasicBlock, [1, 1, 1, 1], input_mode = 'only_input_SIM_images', LR_highway=False,input_nc = num_raw_SIMdata, pretrained=False, progress=False,)
        # SIMnet = UNet(17,1)
        SIMnet.apply(init_weights)
        start_time = time.time()
        train_loss, valid_loss = train(SIMnet, SIM_train_dataloader, SIM_valid_dataloader, criterion, num_epochs, batch_size, device, lr,weight_decay,logfile_directory=None)
        # SIMnet.to('cpu')

        if valid_loss < min_loss:
            best_hyperparams = random_params
            min_loss = valid_loss
        end_time = time.time()

        print(
            'avg train rmse: %f, avg valid rmse: %f learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
            % (train_loss, valid_loss, lr, batch_size, weight_decay,Dropout_ratio, end_time - start_time))
        f_hyparameters.write(
            'avg train rmse: %f, avg valid rmse: %f learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
            % (train_loss, valid_loss, lr, batch_size, weight_decay,Dropout_ratio, end_time - start_time))
        # torch.save(SIMnet.state_dict(), file_directory+ '/SIMnet.pkl')
    f_hyparameters.close()
    print(
        'best_hyperparams : learning_rate:%f, batch_size:%d, weight_decay: %f, '
        % (best_hyperparams['learning_rate'] , best_hyperparams['batch_size'], best_hyperparams['weight_decay']))

    f = open(file_directory + "/best_hyperparams.txt", 'w')
    f.write(str(best_hyperparams))
    f.close()

    # a = SIM_train_dataset[0]
    # image = a[0]
    # image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    # print(SIMnet(image1))


