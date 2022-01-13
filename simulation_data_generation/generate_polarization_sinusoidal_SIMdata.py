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
import cv2

class polarization_Sinusoidal_pattern(SinusoidalPattern):

    def _init_(self,directory_txt_file=None):
       self.directory_txt_file = directory_txt_file
       super(polarization_Sinusoidal_pattern, self)._init_(self, probability=1)
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
            pattern_illumination_images=self.SinusoidalPattern(TensorImage)
            SIMdata_images += self.SR_image_generator(TensorImage)
            SIMdata_images += self.LR_image_generator(TensorImage)
            SIMdata_images += pattern_illumination_images

        return SIMdata_images

    def SinusoidalPattern(self, image):
        '''
        :param image:  PIL_Image that will be loaded pattern on
        :param NumPhase:  Number of phase
        :return: SinusoidalPatternImage: Image which loaded sinusoidal pattern
        '''
        # resolution = 0.61 * self.EmWaveLength / self.NA
        resolution = 0.5 * self.EmWaveLength / self.NA
        # xx, yy, _, _ = self.GridGenerate(image=torch.rand(7, 7))
        # xx, yy, fx, fy = self.GridGenerate(image)
        SinPatternPIL_Image = []
        SIMdata_PIL_Image = []
        # random_initial_direction_phase = random.random() * 2 * math.pi
        random_initial_direction_phase = 1/12  * math.pi

        txt_directory = SourceFileDirectory + '/SIMdata_SR_train/illumination_parameters.txt'
        f = open(txt_directory, 'a')

        for i in range(3):
            modulation_factor = random.random() / 2 + 0.5
            # modulation_factor = 0.05
            theta = i * 2 / 3 * math.pi + random_initial_direction_phase
            SpatialFrequencyX = self.pattern_frequency_ratio * 1 / resolution * math.sin(
                theta)  # 0.8倍的极限频率条纹 pattern_frequency_ratio，可调
            SpatialFrequencyY = self.pattern_frequency_ratio * 1 / resolution * math.cos(theta)
            # random_initial_phase = 1 / 4 * math.pi
            random_initial_phase = random.random() * 2 * math.pi
            SpatialFrequencyX_pixel = SpatialFrequencyX / self.delta_fx
            SpatialFrequencyY_pixel = SpatialFrequencyY / self.delta_fy
            for j in range(self.NumPhase):
                phase = j * 2 * math.pi / self.NumPhase + random_initial_phase
                SIMdata_OTF_filter_gaussian_noise_PIL, SinPattern_PIL,absorption_efficiency = self.generate_single_polarization_SIM_data(image,
                                                                                                      SpatialFrequencyX,
                                                                                                      SpatialFrequencyY,
                                                                                                      modulation_factor,
                                                                                                      phase,theta,polarization='None')

                SIMdata_PIL_Image.append(SIMdata_OTF_filter_gaussian_noise_PIL)
                SinPatternPIL_Image.append(SinPattern_PIL)
                save_name = SourceFileDirectory+'/SIMdata_SR_train/'+str(i*self.NumPhase+j) + "absorption_efficiency.pt"

                torch.save(absorption_efficiency,save_name)
                f.write(str(SpatialFrequencyX_pixel) + '\t' + str(SpatialFrequencyY_pixel) + '\t' + str(modulation_factor) + '\t' + str(phase) + '\n')
        f.close()


        return SIMdata_PIL_Image + SinPatternPIL_Image

    def generate_single_polarization_SIM_data(self,image,SpatialFrequencyX,SpatialFrequencyY,modulation_factor,phase,theta,polarization = 'None'):
        xx, yy,_,_ = self.GridGenerate(up_sample= self.upsample, grid_mode='real')
        SinPattern = (torch.cos(
            phase + 2 * math.pi * (
                    SpatialFrequencyX * xx + SpatialFrequencyY * yy)) * modulation_factor + 1) / 2
        if polarization == 'spiral':
            polarization_angle = torch.atan2(yy,xx)
        elif polarization == 'image':
            camera_man_np = cv2.imread('/data/zh/self_supervised_learning_SR/camera_man.png', -1)/1.0
            camera_man_tensor = torch.from_numpy(camera_man_np)
            polarization_angle = camera_man_tensor
            polarization_angle = polarization_angle/polarization_angle.max() * math.pi

        # polarization_angle = torch.rand_like(xx)

        if polarization == 'None':
            absorption_efficiency = torch.ones_like(xx)
        else:
            absorption_efficiency = 1 - 0.4 * torch.cos(2 * (theta - polarization_angle))
        if self.upsample == False:
            OTF = self.OTF
        else:
            OTF = self.OTF_upsmaple
            up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
            image = up_sample(image.unsqueeze(0))


        SIMdata_OTF_filter = self.OTF_Filter(SinPattern * image * absorption_efficiency, OTF)
        SIMdata_OTF_filter_gaussian_noise = self.add_gaussian_noise(SIMdata_OTF_filter)
        SIMdata_OTF_filter_gaussian_noise = SIMdata_OTF_filter_gaussian_noise.float()
        if self.upsample == True:
            AvgPooling = torch.nn.AvgPool2d(kernel_size= 2)
            SIM_data = AvgPooling(SIMdata_OTF_filter_gaussian_noise.unsqueeze(0).unsqueeze(0))
            SIM_data = SIM_data.squeeze()
            absorption_efficiency = AvgPooling(absorption_efficiency.unsqueeze(0).unsqueeze(0))
            absorption_efficiency = absorption_efficiency.squeeze()
        else:
            SIM_data = SIMdata_OTF_filter_gaussian_noise
        SIMdata_OTF_filter_gaussian_noise_PIL = transforms.ToPILImage()(SIM_data).convert('RGB')
        SinPattern = SinPattern.float()
        SinPattern_PIL = transforms.ToPILImage()(SinPattern).convert('RGB')
        absorption_efficiency_PIL = transforms.ToPILImage()(absorption_efficiency/absorption_efficiency.max()).convert('RGB')
        # polarization_angle_PIL = transforms.ToPILImage()(polarization_angle + polarization_angle.min()).convert('RGB')


        return SIMdata_OTF_filter_gaussian_noise_PIL, absorption_efficiency_PIL, absorption_efficiency




if __name__ == '__main__':
    data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
    SourceFileDirectory = data_generation_parameters['SourceFileDirectory']
    train_directory = SourceFileDirectory  + '/train'
    valid_directory = SourceFileDirectory  + '/valid'
    NumPhase = data_generation_parameters['NumPhase']

    data_num = data_generation_parameters['data_num']
    image_size = data_generation_parameters['image_size']

    p = Pipeline_speckle.Pipeline_speckle(source_directory=train_directory, output_directory="../SIMdata_SR_train")
    p.add_operation(polarization_Sinusoidal_pattern(probability=1))
    p.sample(1,multi_threaded=False,data_type='train',data_num = 3*NumPhase)

