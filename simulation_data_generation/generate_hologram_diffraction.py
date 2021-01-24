#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/22
import torch
import math
from torchvision import transforms
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from utils import common_utils
import numpy as np
from simulation_data_generation import Pipeline_speckle
from numpy.fft import fft2
from numpy.fft import fftshift
from numpy.fft import ifft2
from numpy.fft import ifftshift
from Augmentor.Operations import Crop
from configparser import ConfigParser
from Augmentor import Operations


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

    def expjk_to_cosk_sink_stack(self,k):
        if k.dim() == 2:
            result = torch.stack([torch.cos(k), torch.sin(k)], 2)
        elif k.dim() == 1:
            result = torch.stack([torch.cos(k).unsqueeze(0), torch.sin(k).unsqueeze(0)], 2)
        else:
            raise Exception('The dim of input k must be 1 or 2')
        return result

    def cosk_sink_stack_to_phasek(self,cosk_sink_stack):

        phasek = torch.atan2(cosk_sink_stack[:,:,1],cosk_sink_stack[:,:,0])

        return phasek

    def hologram_diffraction(self, phase_image, data_num):

        phase_image = phase_image.squeeze()
        size_x = phase_image.size()[0]
        size_y = phase_image.size()[1]

        k = torch.tensor([self.wave_num],dtype = torch.float32)
        distance = torch.tensor([self.distance],dtype = torch.float32)
        lamda = torch.tensor([self.WaveLength],dtype = torch.float32)
        # sample_rate = size_x
        fresnel_propagate_intensity_stack = []
        phase_image_complex_amplitude = self.expjk_to_cosk_sink_stack(phase_image)
        xx0, yy0 = self.xx, self.yy
        xx1, yy1 = xx0, yy0

        for i in range(data_num):
            d = (i+1) * distance
            F0_temp = self.complex_divide (self.expjk_to_cosk_sink_stack(k*d), self.expjk_to_cosk_sink_stack(lamda * d))
            F0 =  self.complex_times( F0_temp, self.expjk_to_cosk_sink_stack( k /2 /d * (xx1 * xx1 + yy1 * yy1)))
            F = self.expjk_to_cosk_sink_stack( k /2 /d * (xx0 * xx0 + yy0 * yy0))
            phase_image_complex_amplitude = self.complex_times(phase_image_complex_amplitude, F)
            Ff = self.torch_2d_fftshift(torch.fft(phase_image_complex_amplitude , 2))
            Fuf = self.complex_times(F0, Ff)
            I = pow(Fuf[:,:,0],2) + pow(Fuf[:,:,1],2)

            fresnel_propagate_intensity_normalized = I / I.max()
            fresnel_propagate_intensity_normalized_PIL = transforms.ToPILImage()(fresnel_propagate_intensity_normalized).convert('RGB')
            fresnel_propagate_intensity_stack.append(fresnel_propagate_intensity_normalized_PIL)
            common_utils.plot_single_tensor_image(fresnel_propagate_intensity_normalized)

        return fresnel_propagate_intensity_stack + fresnel_propagate_intensity_stack

    # 角谱算法，由于采样太多，算得太慢需要改进
    # def hologram_diffraction(self, phase_image, data_num):
    #     phase_image = phase_image.squeeze()
    #     size_x = phase_image.size()[0]
    #     size_y = phase_image.size()[1]
    #     real_part_phase_image = torch.cos(phase_image)
    #     imaginary_part_phase_image = torch.sin(phase_image)
    #     phase_image_complex_amplitude = torch.stack([real_part_phase_image,imaginary_part_phase_image],2)
    #     phase_image_spectrum_complex_amplitude = self.torch_2d_fftshift(torch.fft(phase_image_complex_amplitude,2))
    #     # self.xx * self.fx + self.yy * self.fy
    #     fxfx, fyfy, xx, yy = self.fx, self.fy, self.xx, self.yy
    #     wave_length = self.WaveLength
    #     wave_num = self.wave_num
    #     distance = self.distance
    #     complex_image_in_distance = torch.zeros_like(phase_image_spectrum_complex_amplitude)
    #     intensity_image_data_stack = []
    #     speckle_pattern_stack = []
    #     for epoch in range(self.data_num):
    #         for i in range(self.image_size):
    #             for j in range(self.image_size):
    #                 spectrum_at_fxy_complex_amplitude = phase_image_spectrum_complex_amplitude[i,j,:]
    #                 fx = fxfx[i,j]
    #                 fy = fyfy[i, j]
    #                 real_part_xy = torch.cos(math.pi * 2 * (fx * xx + fy * yy))
    #                 imaginary_part_xy = torch.sin(math.pi * 2 * (fx * xx + fy * yy))
    #                 phase_z = (epoch+1) * distance * wave_num * torch.pow(1 - wave_length * wave_length * pow(fx,2) - wave_length * wave_length * pow(fy,2),1/2)
    #                 real_part_z = torch.cos(phase_z)
    #                 imaginary_part_z = torch.sin(phase_z)
    #                 complex_amplitude_xyz = self.complex_times([real_part_xy,imaginary_part_xy],[real_part_z,imaginary_part_z])
    #                 complex_image_in_distance_of_fxy = self.complex_times([spectrum_at_fxy_complex_amplitude[0],spectrum_at_fxy_complex_amplitude[1]],[complex_amplitude_xyz[:,:,0],complex_amplitude_xyz[:,:,1]])
    #                 complex_image_in_distance += complex_image_in_distance_of_fxy
    #                 print(complex_image_in_distance.max())
    #         # speckle_tensor = torch.from_numpy(abs(speckle_intensity))
    #         intensity_image_in_distance = pow(pow(complex_image_in_distance[:,:,0],2) + pow(complex_image_in_distance[:,:,1],2),1/2)
    #         intensity_image_in_distance_normalized = intensity_image_in_distance / intensity_image_in_distance.max() *256
    #         intensity_image_in_distance_PIL = transforms.ToPILImage()(intensity_image_in_distance_normalized).convert('RGB')
    #         intensity_image_data_stack += [intensity_image_in_distance_PIL]
    #         speckle_pattern_stack += [speckle_pattern_stack]
    #
    #     return intensity_image_data_stack + speckle_pattern_stack

    def complex_times(self,a,b):
        real_part_result = a[:,:,0] * b[:,:,0] - a[:,:,1] * b[:,:,1]
        imaginary_part_result = a[:,:,0] * b[:,:,1] + a[:,:,1] * b[:,:,0]
        return torch.stack([real_part_result,imaginary_part_result],2)
    def complex_divide(self,a,b):
        b_temp = torch.zeros_like(b)
        b_temp[:,:,0] = b[:,:,0]
        b_temp[:,:,1] = b[:,:,1] * -1
        numerator  = self.complex_times(a,b_temp)
        denominator = pow(b[:,:,0],2) + pow(b[:,:,1],2)
        if denominator.dim() == 2:
            denominator = denominator.unsqueeze(2)
        result = numerator / denominator
        return result
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

    def torch_2d_fftshift(self,image):
        if not image.size()[2] == 2:
            raise Exception('The channel of input must be 2 -- real_part and imaginary_part')
        image = image.squeeze()
        image_real_part = image[:, :, 0]
        image_imaginary_part = image[:, :, 1]
        if image_real_part.dim() == 2:
            shift = [image_real_part.shape[ax] // 2 for ax in range(image_real_part.dim())]
            image_real_part_shift = torch.roll(image_real_part, shift, [0, 1])
            image_imaginary_part_shift = torch.roll(image_imaginary_part, shift, [0, 1])
        else:
            raise Exception('The dim of image must be 2')

        return torch.stack([image_real_part_shift, image_imaginary_part_shift], 2)

    def torch_2d_ifftshift(self,image):
        if not image.size()[2] == 2:
            raise Exception('The channel of input must be 2 -- real_part and imaginary_part')
        image = image.squeeze()
        image_real_part = image[:, :, 0]
        image_imaginary_part = image[:, :, 1]
        if image_real_part.dim() == 2:
            shift = [-(image.shape[1 - ax] // 2) for ax in range(image_imaginary_part.dim())]
            image_real_part_shift = torch.roll(image_real_part, shift, [1, 0])
            image_imaginary_part_shift = torch.roll(image_imaginary_part, shift, [1, 0])
        else:
            raise Exception('The dim of image must be 2')
        return torch.stack([image_real_part_shift, image_imaginary_part_shift], 2)


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
