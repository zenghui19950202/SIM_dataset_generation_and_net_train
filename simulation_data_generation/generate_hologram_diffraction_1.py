#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/22
import torch
import math
from torchvision import transforms
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from utils import *
import numpy as np
from simulation_data_generation import Pipeline_speckle
from numpy.fft import fft2
from numpy.fft import fftshift
from numpy.fft import ifft2
from numpy.fft import ifftshift
from Augmentor.Operations import Crop
from configparser import ConfigParser
from Augmentor import Operations
from self_supervised_learning_sr import forward_model


class hologram(Operations.Operation):

    def __init__(self, probability, image_size=None):  # unit: nm
        """
        """
        config = ConfigParser()
        if config.read('../configuration_hologram.ini') != []:
            config.read('../configuration_hologram.ini')
        elif config.read('configuration_hologram.ini') != []:
            config.read('configuration_hologram.ini')
        else:
            raise Exception('directory of configuration_hologram.ini error')

        train_net_parameters = {}
        train_directory_file = config.get('image_file', 'SourceFileDirectory') + '/SIMdata_SR_train.txt'
        save_file_directory = config.get('image_file', 'save_file_directory')
        data_num = config.getint('SIM_data_generation', 'data_num')  # the number of raw SIM images
        image_size = config.getint('SIM_data_generation', 'image_size')
        PixelSizeOfCCD = config.getint('SIM_data_generation', 'PixelSizeOfCCD')
        WaveLength = config.getfloat('SIM_data_generation', 'WaveLength')
        distance = config.getint('SIM_data_generation', 'distance')
        SNR = config.getint('SIM_data_generation', 'SNR')

        self.data_num = data_num
        self.image_size = image_size
        self.PixelSize = PixelSizeOfCCD
        self.delta_x = self.PixelSize  # xy方向的空域像素间隔，单位m
        self.delta_y = self.PixelSize
        self.delta_fx = 1 / self.image_size / self.delta_x  # xy方向的频域像素间隔，单位m ^ -1
        self.delta_fy = 1 / self.image_size / self.delta_y
        self.xx, self.yy, self.fx, self.fy = self.GridGenerate(self.image_size)
        self.f = pow((self.fx ** 2 + self.fy ** 2), 1 / 2)  # The spatial freqneucy fr=sqrt( fx^2 + fy^2 )
        self.WaveLength = WaveLength
        self.wave_num = math.pi * 2 / WaveLength
        self.distance = distance
        self.SNR = SNR
        self.xx0,self.xx1,self.yy0,self.yy1 = self.xx,self.xx, self.yy , self.yy


        Operations.Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.

        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        crop_size = self.image_size
        random_speckle_SIMdata_pattern = []
        for image in images:
            h, w = image.size
            pad_w = max(crop_size - w, 0)
            pad_h = max(crop_size - h, 0)
            img_pad = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(image)
            center_crop = transforms.CenterCrop(size=(crop_size, crop_size))
            imag_pad_crop = center_crop(img_pad)

            # augmented_images += self.LR_image_generator(imag_pad_crop)
            image_gray = imag_pad_crop.convert('L')
            TensorImage = transforms.ToTensor()(image_gray)
            random_speckle_SIMdata_pattern += [transforms.ToPILImage()(TensorImage).convert('RGB')]
            random_speckle_SIMdata_pattern += [transforms.ToPILImage()(TensorImage).convert('RGB')]
            random_speckle_SIMdata_pattern += self.hologram_diffraction(TensorImage, self.data_num)

            # random_speckle_SIMdata_pattern.append(self.random_speckle_pattern_SIMdata_generate(TensorImage))
        return random_speckle_SIMdata_pattern

    def hologram_diffraction(self, phase_image, data_num):

        phase_image = phase_image.squeeze()
        size_x = phase_image.size()[0]
        size_y = phase_image.size()[1]

        k = torch.tensor([self.wave_num],dtype = torch.float32)
        distance = torch.tensor([self.distance],dtype = torch.float32)
        lamda = torch.tensor([self.WaveLength],dtype = torch.float32)
        # sample_rate = size_x
        fresnel_propagate_intensity_stack = []
        # phase_image_complex_amplitude = self.expjk_to_cosk_sink_stack(phase_image)
        xx0, yy0 = self.xx, self.yy
        xx1, yy1 = xx0, yy0

        for i in range(data_num):
            d = (i+1) * distance

            diffraction_phase_image = forward_model.fresnel_propagate(phase_image, d, xx0, yy0, xx1, yy1, lamda, k)

            estimated_diffraction_hologram_intensity = pow(diffraction_phase_image[:, :, 0], 2) + pow(
                diffraction_phase_image[:, :, 1], 2)

            fresnel_propagate_intensity_normalized = estimated_diffraction_hologram_intensity / estimated_diffraction_hologram_intensity.max()
            fresnel_propagate_intensity_normalized = self.add_gaussian_noise(fresnel_propagate_intensity_normalized)
            fresnel_propagate_intensity_normalized_PIL = transforms.ToPILImage()(fresnel_propagate_intensity_normalized).convert('RGB')
            fresnel_propagate_intensity_stack.append(fresnel_propagate_intensity_normalized_PIL)
            # common_utils.plot_single_tensor_image(fresnel_propagate_intensity_normalized)

        return fresnel_propagate_intensity_stack + fresnel_propagate_intensity_stack

    def add_gaussian_noise(self, tensor_Image):
        # if len(TensorImage)==3:
        #      TensorImage = TensorImage.permute(1, 2, 0) # transope for matplot
        # signal_intensity_of_image = (tensor_Image ** 2).mean()  # The mean intensity of signal
        signal_std_of_image = (tensor_Image ** 2).std()  # The std intensity of signal
        noise_std_of_image = signal_std_of_image / self.SNR
        noise_of_image = torch.zeros_like(tensor_Image)
        # std_of_noise = noise_std_of_image ** (0.5)
        noise_of_image.normal_(mean=0, std=noise_std_of_image)
        image_with_noise = tensor_Image + noise_of_image
        image_with_noise = torch.where(image_with_noise < 0, torch.zeros_like(image_with_noise), image_with_noise)
        image_with_noise_normalized = image_with_noise / image_with_noise.max()
        return image_with_noise_normalized
        # return image_with_noise_normalized

    def GridGenerate(self, image_size, grid_mode='real'):
        '''
        :param Magnification: the magnification of the Microscope
        :param PixelSize: the PixleSize of the sCMOS or CCD
        :param EmWaveLength:  emission wavelength of sample
        :param NA:  NA(numerical aperture) of the objective
        :return:
        '''
        y, x = image_size, image_size
        if x % 2 == 1:
            if y % 2 == 1:
                xx, yy = torch.meshgrid(torch.arange(-(x - 1) / 2, (x + 1) / 2, 1),
                                        torch.arange(-(y - 1) / 2, (y + 1) / 2, 1))  # 空域x方向坐标为奇数，y方向坐标为奇数的情况
            else:
                xx, yy = torch.meshgrid(torch.arange(-(x - 1) / 2, (x + 1) / 2, 1),
                                        torch.arange(-y / 2, y / 2 - 1, 1))  # 空域x方向坐标为奇数，y方向坐标为偶数的情况
        else:
            if y % 2 == 1:
                xx, yy = torch.meshgrid(torch.arange(-x / 2, x / 2, 1),
                                        torch.arange(-(y - 1) / 2, (y + 1) / 2, 1))  # 空域x方向坐标为偶数，y方向坐标为奇数的情况
            else:
                xx, yy = torch.meshgrid(torch.arange(-x / 2, x / 2, 1),
                                        torch.arange(-y / 2, y / 2, 1))  # 空域x方向坐标为偶数，y方向坐标为偶数的情况

        if grid_mode == 'real':
            fx = xx * self.delta_fx
            fy = yy * self.delta_fy
            xx = xx * self.delta_x
            yy = yy * self.delta_y
        elif grid_mode == 'pixel':
            fx = xx * 1.0 / self.image_size
            fy = yy * 1.0 / self.image_size
        else:
            raise Exception('error grid mode')

        return xx, yy, fx, fy

if __name__ == '__main__':
    config = ConfigParser()
    config.read('../configuration_hologram.ini')
    config.sections()
    SourceFileDirectory = config.get('image_file', 'SourceFileDirectory')
    sample_num_train = config.getint('SIM_data_generation', 'sample_num_train')
    sample_num_valid = config.getint('SIM_data_generation', 'sample_num_valid')
    image_size = config.getint('SIM_data_generation', 'image_size')
    data_num = config.getint('SIM_data_generation', 'data_num')
    # SourceFileDirectory = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/test2"

    train_directory = SourceFileDirectory + '/train'
    valid_directory = SourceFileDirectory + '/valid'

    p = Pipeline_speckle.Pipeline_speckle(source_directory=train_directory, output_directory="../SIMdata_SR_train")
    p.add_operation(Crop(probability=1, width=image_size, height=image_size, centre=False))
    p.add_operation(hologram(probability=1))
    p.sample(sample_num_train, multi_threaded=False, data_type='train', data_num=data_num)
