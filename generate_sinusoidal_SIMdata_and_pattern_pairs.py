#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/10

import torch
import math
import Pipeline_SIMdata_pattern_pairs
from torchvision import transforms
from fuctions_for_generate_pattern import SinusoidalPattern
from Augmentor.Operations import Crop
from configparser import ConfigParser
import random

class sinusoidal_SIMdata_pattern_pair(SinusoidalPattern):

    def _init_(self,directory_txt_file=None):
       self.directory_txt_file = directory_txt_file
       super(sinusoidal_SIMdata_pattern_pair,self)._init_(self, probability=1, NumPhase=3, Magnification=150, PixelSizeOfCCD=6800,
                                      EmWaveLength=635, NA=0.9, SNR=500,
                                      image_size=256)
    def perform_operation(self, images):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.

        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        crop_size=self.image_size
        image_data_pair=[]
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

            image_data_pair = self.SinusoidalPattern(TensorImage)

        return image_data_pair

    def SinusoidalPattern(self, image):
        '''
        :param image:  PIL_Image that will be loaded pattern on
        :param NumPhase:  Number of phase
        :return: SinusoidalPatternImage: Image which loaded sinusoidal pattern
        '''
        resolution = 0.61 * self.EmWaveLength / self.NA
        # xx, yy, _, _ = self.GridGenerate(image=torch.rand(7, 7))
        # xx, yy, fx, fy = self.GridGenerate(image)
        # TensorImage = transforms.ToTensor()(image)

        random_theta = random.random() * 2 * math.pi
        SpatialFrequencyX = -self.pattern_frequency_ratio * 1 / resolution * math.sin(random_theta)  # 0.8倍的极限频率条纹 pattern_frequency_ratio，可调
        SpatialFrequencyY = -self.pattern_frequency_ratio * 1 / resolution * math.cos(random_theta)

        random_phase = random.random() * 2 * math.pi
        random_modulation = random.random()/2 +0.5
        SinPattern = (torch.cos(
            random_phase + 2 * math.pi * (SpatialFrequencyX * self.xx + SpatialFrequencyY * self.yy)) * random_modulation + 1) / 2
        SinPattern_OTF_filter = self.OTF_Filter(SinPattern * image,self.OTF)
        SinPattern_OTF_filter_gaussian_noise = self.add_gaussian_noise(SinPattern_OTF_filter)

        SinPattern_OTF_filter_gaussian_noise_PIL = transforms.ToPILImage()(SinPattern_OTF_filter_gaussian_noise).convert('RGB')
        SinPattern_PIL_ = transforms.ToPILImage()(SinPattern).convert('RGB')

        return [SinPattern_OTF_filter_gaussian_noise_PIL,SinPattern_PIL_]


if __name__ == '__main__':

    config = ConfigParser()
    config.read('configuration.ini')
    config.sections()
    SourceFileDirectory = config.get('image_file', 'SourceFileDirectory')
    sample_num = config.getint('SIM_data_generation', 'sample_num')
    image_size = config.getint('SIM_data_generation', 'image_size')
    data_ratio = config.getfloat('SIM_data_generation', 'data_ratio')
    # SourceFileDirectory = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/test2"

    # p = Pipeline_speckle.Pipeline_speckle(source_directory=SourceFileDirectory)
    # p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    # p.add_operation(SinusoidalPattern(probability=1,image_size=image_size))
    # p.sample(20,multi_threaded=True,data_ratio=1)

    train_directory = SourceFileDirectory + '/train'
    valid_directory = SourceFileDirectory + '/valid'

    p = Pipeline_SIMdata_pattern_pairs.Pipeline_SIMdata_pattern_pairs(source_directory=train_directory,output_directory="train")
    p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    p.add_operation(sinusoidal_SIMdata_pattern_pair(probability=1,image_size=image_size))
    p.sample(10,multi_threaded=True,data_ratio=1)

    # p = Pipeline_speckle.Pipeline_speckle(source_directory=valid_directory,output_directory="valid")
    # p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    # p.add_operation(SinusoidalPattern(probability=1,image_size=image_size))
    # p.sample(400,multi_threaded=True,data_ratio=0)
