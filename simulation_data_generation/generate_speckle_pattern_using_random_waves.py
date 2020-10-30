#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/16
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/10

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

class random_sepckle_pattern(SinusoidalPattern):

    def _init_(self, directory_txt_file=None):
        self.directory_txt_file = directory_txt_file
        super(random_sepckle_pattern, self)._init_(self, probability=1)

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
            random_speckle_SIMdata_pattern += self.SR_image_generator(TensorImage)
            random_speckle_SIMdata_pattern += self.LR_image_generator(TensorImage)
            random_speckle_SIMdata_pattern += self.random_speckle_pattern_generate(TensorImage,self.data_num)

            # random_speckle_SIMdata_pattern.append(self.random_speckle_pattern_SIMdata_generate(TensorImage))
        return random_speckle_SIMdata_pattern

    def random_speckle_pattern_generate(self, image,data_num):
        shift_pixel_num = math.sqrt(data_num)
        if not int(shift_pixel_num) == shift_pixel_num:
            raise Exception('data_num must be square of integer number')
        else:
            shift_pixel_num = int(shift_pixel_num)
            image = image.squeeze()
            size_x = image.size()[0] + shift_pixel_num
            size_y = image.size()[1] + shift_pixel_num
            experimental_parameters = SinusoidalPattern(probability=1,image_size = size_x)
            CTF = experimental_parameters.CTF
            diffuser_amplitude = np.random.rand(size_x,size_y)
            diffuser_phase =  np.exp(-1j * 2 * math.pi * np.random.rand(size_x,size_y))
            object = diffuser_amplitude * diffuser_phase
            speckle_spectrum = fftshift(fft2(object, axes=(0, 1)), axes=(0, 1))
            speckle_spectrum_low_filtered = speckle_spectrum * CTF.numpy()
            speckle = ifft2(ifftshift(speckle_spectrum_low_filtered, axes=(0, 1)), axes=(0, 1))
            speckle_intensity = speckle * speckle.conj()
            speckle_tensor = torch.from_numpy(abs(speckle_intensity))
            speckle_SIM_data_stack = []
            speckle_pattern_stack = []
            for i in range(shift_pixel_num):
                for j in range(shift_pixel_num):
                    speckle_tensor_crop = speckle_tensor[i:i+image.size()[0],j:j+image.size()[1]]
                    speckle_SIM_data = self.add_gaussian_noise(self.OTF_Filter(speckle_tensor_crop * image, self.OTF))
                    speckle_SIM_data_normalized = speckle_SIM_data/ speckle_SIM_data.max()
                    speckle_SIM_data_normalized = speckle_SIM_data_normalized.float()
                    speckle_SIM_data_normalized_PIL=transforms.ToPILImage()(speckle_SIM_data_normalized).convert('RGB')
                    speckle_SIM_data_stack.append(speckle_SIM_data_normalized_PIL)

                    speckle_tensor_crop_normalized = speckle_tensor_crop.float() / speckle_tensor_crop.max()
                    speckle_tensor_crop_normalized_PIL = transforms.ToPILImage()(speckle_tensor_crop_normalized).convert('RGB')
                    speckle_pattern_stack.append(speckle_tensor_crop_normalized_PIL)
        return speckle_SIM_data_stack + speckle_pattern_stack

    # def random_speckle_pattern_generate(self,images):
    #     '''
    #     :param image:  PIL_Image that will be loaded pattern on
    #     :param NumPhase:  Number of phase
    #     :return: SinusoidalPatternImage: Image which loaded sinusoidal pattern
    #     '''
    #     resolution = 0.61 * self.EmWaveLength / self.NA
    #     wave_nums = 1000
    #     random_phi = torch.rand(1, 1, wave_nums) * 1 / 2 * math.pi
    #     random_theta = torch.rand(1, 1, wave_nums) * 2 * math.pi
    #     random_initial_phase = torch.rand(1, 1, wave_nums) * 2 * math.pi
    #
    #     kxy = 1 / resolution * torch.cos(random_theta)
    #     kx = kxy * torch.sin(random_phi)
    #     ky = kxy * torch.cos(random_phi)
    #     xx = self.xx.view(self.image_size,self.image_size,1)
    #     yy = self.yy.view(self.image_size,self.image_size,1)
    #     phase_pattern = torch.exp(1j * random_initial_phase + 2 * math.pi * (kx * xx + ky * yy))
    #     phase_pattern_sum = torch.sum(phase_pattern)
    #     random_sepckle_pattern = phase_pattern_sum * phase_pattern_sum.conjugate()
    #     random_sepckle_SIMdata = random_sepckle_pattern * images
    #     return random_sepckle_SIMdata

if __name__ == '__main__':

    # data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
    # train_directory = data_generation_parameters['SourceFileDirectory']  + '/train'
    # valid_directory = data_generation_parameters['SourceFileDirectory']  + '/valid'
    #
    # data_num = data_generation_parameters['data_num']
    # image_size = data_generation_parameters['image_size']
    # sample_num_train = data_generation_parameters['sample_num_train']
    #
    # p = Pipeline_speckle.Pipeline_speckle(source_directory=train_directory, output_directory="../SIMdata_SR_train")
    # p.add_operation(random_sepckle_pattern(probability = 1))
    # p.sample(sample_num_train, multi_threaded=False, data_type='train', data_num=data_num)

    # a = random_sepckle_pattern(probability=1)
    # resolution = 635 / 0.9
    # wave_nums = 1000
    # #伪随机的机制能否会对仿真结果造成影响。
    # #写一个数据生成的pipeli
    # random_phi = torch.rand(1, 1, wave_nums) * 1 / 2 * math.pi
    # random_theta = torch.rand(1, 1, wave_nums) * 2 * math.pi
    # random_initial_phase = torch.rand(1, 1, wave_nums) * 2 * math.pi
    #
    # kxy = 1 / resolution * torch.cos(random_theta)
    # kx = kxy * torch.sin(random_phi)
    # ky = kxy * torch.cos(random_phi)
    # xx = a.xx.view(a.image_size, a.image_size, 1)
    # yy = a.yy.view(a.image_size, a.image_size, 1)
    # phase_pattern_real = torch.cos(random_initial_phase + 2 * math.pi * (kx * xx + ky * yy))
    # phase_pattern_imag = torch.sin(random_initial_phase + 2 * math.pi * (kx * xx + ky * yy))
    # phase_pattern_real_sum = torch.sum(phase_pattern_real, dim=2)
    # phase_pattern_imag_sum = torch.sum(phase_pattern_imag, dim=2)
    # random_sepckle_pattern = torch.pow(torch.pow(phase_pattern_real_sum,2) + torch.pow(phase_pattern_imag_sum,2),1/2)
    #
    # random_sepckle_pattern = random_sepckle_pattern/random_sepckle_pattern.max()
    # common_utils.plot_single_tensor_image(random_sepckle_pattern)
    # random_sepckle_pattern_PIL = transforms.ToPILImage()(random_sepckle_pattern).convert('RGB')
    # random_sepckle_pattern_PIL.save('/home/common/Zenghui/speckle/speckle_pattern.png')
    data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
    train_directory = data_generation_parameters['SourceFileDirectory']  + '/train'
    valid_directory = data_generation_parameters['SourceFileDirectory']  + '/valid'

    data_num = data_generation_parameters['data_num']
    image_size = data_generation_parameters['image_size']

    p = Pipeline_speckle.Pipeline_speckle(source_directory=train_directory, output_directory="../SIMdata_SR_train")
    p.add_operation(random_sepckle_pattern(probability=1))
    p.sample(1,multi_threaded=True,data_type='train',data_num = data_num)
