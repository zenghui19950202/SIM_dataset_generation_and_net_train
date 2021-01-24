#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/18
import torch
from torchvision import transforms
from utils.SpeckleSIMDataLoad import SIM_data_load
from models.Networks_Unet_GAN import UnetGenerator
from torch.utils.data import DataLoader
from simulation_data_generation import SRimage_metrics
from configparser import ConfigParser
from models import *
# from utils import common_utils
from utils import *
import os

def result_net_valiated(SIMnet, input_SIM_images, HR_LR_image, output_dir,normalize = True,):
    LR_image = HR_LR_image[:, :, 1]
    HR_image = HR_LR_image[:, :, 0]
    data_num = config.getint('SIM_data_generation', 'data_num')
    image_size = input_SIM_images.size()
    normalized_image_minibatch = input_SIM_images.view([1, image_size[0], image_size[1], image_size[2]])
    mean_image = normalized_image_minibatch[:, data_num, :, :]

    # mean_image = mean_image* 0.5 + 0.5
    # LR_image = LR_image * 0.5 + 0.5
    # HR_image = HR_image * 0.5 + 0.5

    result_image = SIMnet(normalized_image_minibatch)
    result_image = result_image.squeeze()

    PSNR = SRimage_metrics.calculate_psnr(result_image, HR_image)
    PSNR_LR_HR = SRimage_metrics.calculate_psnr(LR_image, HR_image)
    PSNR_mean_HR = SRimage_metrics.calculate_psnr(mean_image, HR_image)
    print('PSNR:%f , PSNR_LR_HR: %f, PSNR_mean_HR: %f' % (PSNR, PSNR_LR_HR, PSNR_mean_HR))

    result_image = result_image.cpu()
    if normalize == True:
        result_image = result_image * 0.5 + 0.5
    result_image = abs(result_image / result_image.max())
    # image_PIL = transforms.ToPILImage()(result_image).convert('RGB')
    # image_PIL.show()
    common_utils.plot_single_tensor_image(result_image)
    common_utils.save_image_tensor2pillow(result_image.unsqueeze(0).unsqueeze(0),output_dir)
    # image = HR_LR_image[:, :, 1] * 0.5 + 0.5
    # image_PIL = transforms.ToPILImage()(image).convert('RGB')
    # image_PIL.show()
    #
    # image = normalized_image_minibatch[:, data_num, :, :] * 0.5 + 0.5
    # image_PIL = transforms.ToPILImage()(image).convert('RGB')
    # image_PIL.show()


def evaluate_valid_loss(data_iter, criterion, net, device=torch.device('cpu')):
    net.eval()
    """Evaluate accuracy of a model on the given data set."""
    loss_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            loss_sum += criterion(y_hat, y)
            n += y.shape[0]
    return loss_sum.item() / n


if __name__ == '__main__':
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    data_generate_mode = train_net_parameters['data_generate_mode']
    net_type = train_net_parameters['net_type']
    data_input_mode = train_net_parameters['data_input_mode']
    LR_highway_type = train_net_parameters['LR_highway_type']
    data_num = train_net_parameters['data_num']

    config = ConfigParser()
    config.read('configuration.ini')
    SourceFileDirectory = config.get('image_file', 'verification_file_directory')

    num_raw_SIMdata, output_nc, num_downs = data_num, 1, 5

    SIMnet = resnet_backbone_net._resnet('resnet34', resnet_backbone_net.BasicBlock, [1, 1, 1, 1],
                                         input_mode=data_input_mode, LR_highway=LR_highway_type,
                                         input_nc=num_raw_SIMdata,
                                         pretrained=False, progress=False, )
    # SIMnet = UnetGenerator(num_raw_SIMdata, output_nc, num_downs, ngf=64, LR_highway=LR_highway_type, input_mode=data_input_mode,
    #                        use_dropout=False)
    # SIMnet.load_state_dict(torch.load('F:\PHD\SIMDataSet/SIMnet.pkl'))
    # SIMnet = UNet(num_raw_SIMdata, 1, input_mode=data_input_mode, LR_highway=LR_highway_type)
    SIMnet.load_state_dict(torch.load(
        '/home/zh/self_supervised_learning_SR/train_result/resnet_SIM_and_sum_images_only_input_SIM_images_add_11_09/lr_0.0003num_epochs_200batch_size_32weight_decay_1e-05/SIMnet.pkl',map_location='cuda:7'))

    # train_directory_file = SourceFileDirectory + '/SIMdata_pattern_pairs_train.txt'
    valid_directory_file = "D:\DataSet\DIV2K/test"+ '/SIMdata_SR_train.txt'
    valid_directory_file = "/home/common/zenghui/microtube1_512/" + '/SIMdata_SR_train.txt'
    valid_directory_file = SourceFileDirectory + '/SIMdata_SR_train.txt'
    SIM_valid_dataset = SIM_data_load(valid_directory_file,data_mode = data_generate_mode,normalize = True)

    # criterion = criterion = nn.MSELoss()
    SIM_valid_dataloader = DataLoader(SIM_valid_dataset, batch_size=1, shuffle=True)
    # valid_loss = evaluate_valid_loss(SIM_valid_dataloader, criterion, SIMnet, device=torch.device('cpu'))
    # print('valid_loss:%f',valid_loss)
    a, b = SIM_valid_dataset[0]
    output_dir = os.path.dirname(valid_directory_file)
    result_net_valiated(SIMnet, a, b,output_dir = output_dir,normalize = True,)
