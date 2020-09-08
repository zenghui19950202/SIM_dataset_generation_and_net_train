#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/18
import torch
from torchvision import transforms
from SpeckleSIMDataLoad import SIM_data
from Networks_Unet_GAN import UnetGenerator
import torch.nn as nn
from torch.utils.data import DataLoader
import resnet_backbone_net as res_SIMnet
import SRimage_metrics

def result_net_valiated(SIMnet,input_SIM_images,HR_LR_image):
    image_size = input_SIM_images.size()
    normalized_image_minibatch=input_SIM_images.view([1,image_size[0],image_size[1],image_size[2]])
    result_image = SIMnet(normalized_image_minibatch)
    result_image = result_image.squeeze()
    PSNR = SRimage_metrics.calculate_psnr(result_image,HR_LR_image[:,:,0])
    PSNR_LR_HR = SRimage_metrics.calculate_psnr(HR_LR_image[:, :, 1], HR_LR_image[:, :, 0])
    PSNR_mean_HR = SRimage_metrics.calculate_psnr(normalized_image_minibatch[:,16,:,:], HR_LR_image[:, :, 0])
    print('PSNR:%f , PSNR_LR_HR: %f, PSNR_mean_HR: %f' %(PSNR,PSNR_LR_HR,PSNR_mean_HR))
    image = result_image * 0.5 + 0.5
    image_PIL = transforms.ToPILImage()(image).convert('RGB')
    image_PIL.show()

def evaluate_valid_loss(data_iter,criterion, net, device=torch.device('cpu')):
    net.eval()
    """Evaluate accuracy of a model on the given data set."""
    loss_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            y_hat=net(X)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            loss_sum += criterion(y_hat, y)
            n += y.shape[0]
    return loss_sum.item() / n

if __name__ == '__main__':
    input_nc, output_nc, num_downs = 17, 1, 5
    SIMnet = res_SIMnet._resnet('resnet34', res_SIMnet.BasicBlock, [1, 1, 1, 1],input_mode = 'input_SIM_and_sum_images',LR_highway = False, pretrained=False, progress=False)
    # SIMnet = UnetGenerator(input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    # SIMnet.load_state_dict(torch.load('F:\PHD\SIMDataSet/SIMnet.pkl'))
    SIMnet.load_state_dict(torch.load('F:\PHD\SIMnet_train_result\Input_16_plus_mean_no_highway_lr_0.001num_epochs_303batch_size_32weight_decay_1e-050.003970414666192872/SIMnet.pkl'))

    train_directory_file = 'D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/test1/train.txt'
    # valid_directory_file = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/valid.txt"

    SIM_valid_dataset = SIM_data(train_directory_file)

    # criterion = criterion = nn.MSELoss()
    SIM_valid_dataloader = DataLoader(SIM_valid_dataset, batch_size=1, shuffle=True)
    # valid_loss = evaluate_valid_loss(SIM_valid_dataloader, criterion, SIMnet, device=torch.device('cpu'))
    # print('valid_loss:%f',valid_loss)
    a,b = SIM_valid_dataset[0]
    result_net_valiated(SIMnet,a,b)
