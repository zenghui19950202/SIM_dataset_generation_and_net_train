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
            self.random_speckle_pattern_SIMdata_generate(TensorImage)

            random_speckle_SIMdata_pattern.append(self.random_speckle_pattern_SIMdata_generate(TensorImage))
        return random_speckle_SIMdata_pattern

    def random_speckle_pattern_generate(self,images):
        '''
        :param image:  PIL_Image that will be loaded pattern on
        :param NumPhase:  Number of phase
        :return: SinusoidalPatternImage: Image which loaded sinusoidal pattern
        '''
        resolution = 0.61 * self.EmWaveLength / self.NA
        wave_nums = 100
        random_phi = torch.rand(1, 1, wave_nums) * 1 / 2 * math.pi
        random_theta = torch.rand(1, 1, wave_nums) * 2 * math.pi
        random_initial_phase = torch.rand(1, 1, wave_nums) * 2 * math.pi

        kxy = 1 / resolution * torch.cos(random_theta)
        kx = kxy * torch.sin(random_phi)
        ky = kxy * torch.cos(random_phi)
        xx = self.xx.view(self.image_size,self.image_size,1)
        yy = self.yy.view(self.image_size,self.image_size,1)
        phase_pattern = torch.exp(1j * random_initial_phase + 2 * math.pi * (kx * xx + ky * yy))
        phase_pattern_sum = torch.sum(phase_pattern)
        random_sepckle_pattern = phase_pattern_sum * phase_pattern_sum.conjugate()
        random_sepckle_SIMdata = random_sepckle_pattern * images
        return random_sepckle_SIMdata



if __name__ == '__main__':
    a = random_sepckle_pattern(probability = 1)
    resolution = 635 / 0.9
    wave_nums = 500
    #伪随机的机制能否会对仿真结果造成影响。
    #写一个数据生成的pipeli
    random_phi = torch.rand(1, 1, wave_nums) * 1 / 2 * math.pi
    random_theta = torch.rand(1, 1, wave_nums) * 2 * math.pi
    random_initial_phase = torch.rand(1, 1, wave_nums) * 2 * math.pi

    kxy = 1 / resolution * torch.cos(random_theta)
    kx = kxy * torch.sin(random_phi)
    ky = kxy * torch.cos(random_phi)
    xx = a.xx.view(a.image_size, a.image_size, 1)
    yy = a.yy.view(a.image_size, a.image_size, 1)
    phase_pattern_real = torch.cos(random_initial_phase + 2 * math.pi * (kx * xx + ky * yy))
    phase_pattern_imag = torch.sin(random_initial_phase + 2 * math.pi * (kx * xx + ky * yy))

    phase_pattern_real_sum = torch.sum(phase_pattern_real, dim=2)
    phase_pattern_imag_sum = torch.sum(phase_pattern_imag, dim=2)
    random_sepckle_pattern = torch.pow(torch.pow(phase_pattern_real_sum,2) + torch.pow(phase_pattern_imag_sum,2),1/2)
    random_sepckle_pattern_PIL = transforms.ToPILImage()(random_sepckle_pattern).convert('RGB')
    random_sepckle_pattern_PIL.save('D:\DataSet\DIV2K\speckle/speckle_pattern.png')
    random_sepckle_pattern_PIL.show()