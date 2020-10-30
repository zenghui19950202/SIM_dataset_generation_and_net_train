#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/10
from utils import *
import torch
import math
from simulation_data_generation import Pipeline_speckle
from torchvision import transforms
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from Augmentor.Operations import Crop
from configparser import ConfigParser
import random


class multi_spot_pattern(SinusoidalPattern):

    def _init_(self,directory_txt_file=None):
       self.directory_txt_file = directory_txt_file
       super(multi_spot_pattern,self)._init_(self, probability=1)
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
        SIMdata_images=[]
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
            pattern_illumination_images=self.multi_spot_pattern(TensorImage)
            SIMdata_images += self.SR_image_generator(TensorImage)
            SIMdata_images += self.LR_image_generator(TensorImage)
            SIMdata_images += pattern_illumination_images

        return SIMdata_images

    def equal_spacing_vector_generate(self,spacing):
        illunimation_size = self.image_size + spacing*2
        vector = torch.linspace(1 , illunimation_size, illunimation_size)
        vector = torch.fmod(vector, spacing)
        zero_index = (vector == 0).nonzero()
        vector = torch.zeros(illunimation_size, 1)
        vector[zero_index] = 1
        return  vector

    def multi_spot_pattern(self, TensorImage):
        resolution = 0.61 * self.EmWaveLength / self.NA
        PixelSize = self.PixelSizeOfCCD / self.Magnification
        spot_spacing= 2 * math.ceil(resolution/PixelSize)

        vector = self.equal_spacing_vector_generate(spot_spacing)
        multi_spot_location = vector.mm(vector.t()) * 256
        m = torch.nn.ZeroPad2d(spot_spacing)
        Padding_OTF = m(self.OTF)

        multi_spot_pattern = self.OTF_Filter(multi_spot_location, Padding_OTF)

        sample_rate = 4 # sample four points in one psf wide which is fullfilled the sampling theorem
        scan_step = math.ceil(spot_spacing/sample_rate)
        speckle_SIM_data_stack=[]
        initial_i = random.randint(0,3)
        initial_j = random.randint(0,3)
        for i in range(sample_rate):
            for j in range(sample_rate):
                a = (i + initial_i) % sample_rate
                b = (j + initial_j) % sample_rate
                multi_spot_pattern_crop = multi_spot_pattern[
                                          a * scan_step + spot_spacing:a * scan_step + spot_spacing + self.image_size,
                                          b * scan_step + spot_spacing:b * scan_step + spot_spacing + self.image_size]
                speckle_SIM_data = self.add_gaussian_noise(self.OTF_Filter(multi_spot_pattern_crop * TensorImage, self.OTF))
                speckle_SIM_data_normalized = speckle_SIM_data/ speckle_SIM_data.max()

                #TODO: error
                speckle_SIM_data_normalized = speckle_SIM_data_normalized.float()
                speckle_SIM_data_normalized_PIL=transforms.ToPILImage()(speckle_SIM_data_normalized).convert('RGB')
                speckle_SIM_data_stack.append(speckle_SIM_data_normalized_PIL)
                # TODO: complete the speckle SIM data generate code
        return speckle_SIM_data_stack




if __name__ == '__main__':
    data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
    train_directory = data_generation_parameters['SourceFileDirectory']  + '/train'
    valid_directory = data_generation_parameters['SourceFileDirectory']  + '/valid'

    data_num = data_generation_parameters['data_num']
    image_size = data_generation_parameters['image_size']

    p = Pipeline_speckle.Pipeline_speckle(source_directory=train_directory, output_directory="../speckle_SIM_data_train")
    p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    p.add_operation(multi_spot_pattern(probability=1))
    p.sample(1,multi_threaded=True,data_type='train',data_num = 16)

    p = Pipeline_speckle.Pipeline_speckle(source_directory=valid_directory, output_directory="../speckle_SIM_data_valid")
    p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    p.add_operation(multi_spot_pattern(probability=1))
    p.sample(1,multi_threaded=True,data_type='valid',data_num = 16)
