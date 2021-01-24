#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/12/12

from models import *
from utils import *
import torch
import torch.optim as optim
import torch.nn as nn
import random
import time
from utils.SpeckleSIMDataLoad import SIM_data_load
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from early_stopping.pytorchtools import EarlyStopping
from Unet_for_pattern_detection.DataLoad import SIMdata_pattern_pairs
import torchvision.models.resnet as resnet

def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:7')
    else:
        device = torch.device('cpu')
    return device


def train(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None, weight_decay=1e-4,
          logfile_directory=None):
    """Train and evaluate a model with CPU or GPU."""
    writer = SummaryWriter(logfile_directory)
    print('training on', device)
    # nn.DataParallel(net,device_ids=[0,1,2,3])
    # nn.DataParallel(net, device_ids=[0])
    # net = net.to(device)
    net = nn.DataParallel(net)
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5,
                                                           verbose=False, threshold=0.00001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)

    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=False)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    for epoch in range(num_epochs):
        net.train()  # Switch to training mode
        n, start = 0, time.time()
        # train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        # train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_l_sum = torch.tensor([0.0], dtype=torch.float32).cuda()
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32).cuda()


        for X, y in train_iter:
            optimizer.zero_grad()
            # X, y = X.to(device), y.to(device)
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
                n += 1
        train_loss = train_l_sum / n

        valid_loss, PSNR = evaluate_valid_loss(test_iter, criterion, net, device)
        scheduler.step(valid_loss)
        early_stopping(valid_loss, net, epoch)

        writer.add_scalars('scalar/loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch + 1)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('epoch: %d/%d, train_loss: %f, valid_loss: %f ， PSNR ：%f ' % (
        epoch + 1, num_epochs, train_loss, valid_loss, PSNR))
        # print('epoch: %d, train_time: %f, evaluate_time: %f , early_stopping_time: %f ' % (epoch,train_time, evaluate_time,early_stopping_time))
    writer.close
    return train_loss, valid_loss, PSNR


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def evaluate_valid_loss(data_iter, criterion, net, device=torch.device('cpu')):
    net.eval()
    """Evaluate accuracy of a model on the given data set."""
    # loss_sum, PSNR, n, = torch.tensor([0], dtype=torch.float32, device=device), torch.tensor([0], dtype=torch.float32,
    #                                                                                          device=device), 0
    loss_sum, PSNR, n, = torch.tensor([0], dtype=torch.float32).cuda(), torch.tensor([0], dtype=torch.float32).cuda(), 0
    for X, y in data_iter:
        # Copy the data to device.
        # X, y = X.to(device), y.to(device)
        X, y = X.cuda(), y.cuda()
        with torch.no_grad():
            y = y.float()
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            loss_sum += criterion(y_hat, y)
            PSNR += calculate_psnr(y_hat, y)
            n += 1
    return loss_sum / n, PSNR / n


def calculate_psnr(pattern_estimated, pattern_gt):
    pattern_estimated = (pattern_estimated * 0.5 + 0.5)
    pattern_gt = (pattern_gt * 0.5 + 0.5)
    L2_diff = torch.pow((pattern_estimated - pattern_gt), 2)
    MSE = L2_diff.mean()
    pattern_gt_max = pattern_gt.max()
    PSNR = torch.mean(20 * torch.log10(pattern_gt_max / torch.pow(MSE, 1 / 2)))
    return PSNR


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
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

    param_grid = {
        'learning_rate': [0.0001,0.001,0.01],
        'batch_size': [128,256],
        'weight_decay': [1e-5],
        'Dropout_ratio': [1]
    }

    SIM_train_dataset = SIMdata_pattern_pairs(train_directory_file)
    SIM_valid_dataset = SIMdata_pattern_pairs(valid_directory_file)

    random.seed(50)  # 设置随机种子
    min_loss = 1e5

    directory_path = os.path.abspath(os.path.dirname(os.getcwd()))
    file_directory = directory_path + '/train_result/' + net_type + '_' + data_generate_mode + '_' + data_input_mode + '_' + LR_highway_type
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    f_hyparameters = open(file_directory + "/hyperparams.txt", 'w')
    for i in range(MAX_EVALS):
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        lr = random_params['learning_rate']
        batch_size = random_params['batch_size']
        weight_decay = random_params['weight_decay']
        Dropout_ratio = random_params['Dropout_ratio']

        logfile_directory = file_directory + '/' + 'lr_' + str(lr) + 'num_epochs_' + str(
            num_epochs) + 'batch_size_' + str(
            batch_size) + 'weight_decay_' + str(weight_decay)

        device = try_gpu()

        # criterion = MSE_loss()
        criterion = nn.MSELoss()

        SIM_train_dataloader = DataLoader(SIM_train_dataset, num_workers=8, pin_memory=True, batch_size=batch_size,
                                          shuffle=True)
        SIM_valid_dataloader = DataLoader(SIM_valid_dataset, num_workers=8, pin_memory=True, batch_size=batch_size,
                                          shuffle=True)

        num_raw_SIMdata, output_nc, num_downs = 1, 1, 5
        if net_type == 'Unet':
            SIMnet = Networks_Unet_GAN.UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=LR_highway_type,
                                   input_mode=data_input_mode, use_dropout=False)
        elif net_type == 'resnet':
            SIMnet = resnet.resnet101()
        elif net_type == 'wide_Unet':
            SIMnet = Unet_NC2020.UNet(num_raw_SIMdata, 1, input_mode=data_input_mode, LR_highway=LR_highway_type)
        else:
            raise Exception("error net type")
        SIMnet.apply(init_weights)
        start_time = time.time()
        train_loss, valid_loss, PSNR = train(SIMnet, SIM_train_dataloader, SIM_valid_dataloader, criterion, num_epochs,
                                             batch_size, device, lr, weight_decay, logfile_directory=logfile_directory)
        # SIMnet.to('cpu')
        torch.save(SIMnet.state_dict(), logfile_directory + '/SIMnet.pkl')

        if valid_loss < min_loss:
            best_hyperparams = random_params
            min_loss = valid_loss
        end_time = time.time()

        print(
            'avg train rmse: %f, avg valid rmse: %f ,PSNR: %f, learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f '
            % (train_loss, valid_loss, PSNR, lr, batch_size, weight_decay, Dropout_ratio, end_time - start_time))
        f_hyparameters.write(
            'avg train rmse: %f, avg valid rmse: %f ,PSNR: %f, learning_rate:%f, batch_size:%d,weight_decay: %f,Dropout_ratio: %f, time: %f \n '
            % (train_loss, valid_loss, PSNR, lr, batch_size, weight_decay, Dropout_ratio, end_time - start_time))
        # torch.save(SIMnet.state_dict(), file_directory+ '/SIMnet.pkl')
    f_hyparameters.close()
    print(
        'best_hyperparams : learning_rate:%f, batch_size:%d, weight_decay: %f, '
        % (best_hyperparams['learning_rate'], best_hyperparams['batch_size'], best_hyperparams['weight_decay']))

    f = open(file_directory + "/best_hyperparams.txt", 'w')
    f.write(str(best_hyperparams))
    f.close()

    # a = SIM_train_dataset[0]
    # image = a[0]
    # image1 = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    # print(SIMnet(image1))
