#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/3/10

import torch
import math
import numpy as np
from Augmentor import Operations
from torchvision import transforms
import random
from configparser import ConfigParser
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# def AddSinusoidalPattern(pipeline, probability=1):
#     """
#     The function is used to add sinusoidal pattern and OTF on the images in pipeline
#     :param pipeline: The image pipeline based on module 'Augmentor'
#     :param probability: Controls the probability that the operation is
#          performed when it is invoked in the pipeline.
#     :type probability: Float
#     :return: None
#     """
#     if not 0 < probability <= 1:
#         raise ValueError(Pipeline._probability_error_text)
#     else:
#         pipeline.add_operation(SinusoidalPattern(probability=probability))

class SinusoidalPattern(Operations.Operation):
    """
    This class is used to add Sinusoidal Pattern on images.
    """
    def __init__(self,probability):  # unit: nm
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`percentage_area` to control the area
        of the image to crop in terms of its percentage of the original image,
        and a :attr:`centre` parameter toggle whether a random area or the
        centre of the images should be cropped.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        config = ConfigParser()
        config.read('../configuration.ini')
        SourceFileDirectory = config.get('image_file', 'SourceFileDirectory')
        self.Magnification = config.getint('SIM_data_generation', 'Magnification')
        self.PixelSizeOfCCD = config.getint('SIM_data_generation', 'PixelSizeOfCCD')
        self.EmWaveLength = config.getint('SIM_data_generation', 'EmWaveLength')
        self.NA = config.getfloat('SIM_data_generation', 'NA')
        self.NumPhase = config.getint('SIM_data_generation', 'NumPhase')
        self.SNR = config.getint('SIM_data_generation', 'SNR')
        self.image_size = config.getint('SIM_data_generation', 'image_size')
        self.pattern_frequency_ratio = config.getfloat('SIM_data_generation', 'pattern_frequency_ratio')
        self.xx, self.yy, fx, fy = self.GridGenerate(self.image_size)
        self.f0 = 2 * self.NA / self.EmWaveLength  # The coherent cutoff frequency
        self.f = pow((fx ** 2 + fy ** 2), 1 / 2)  # The spatial freqneucy fr=sqrt( fx^2 + fy^2 )

        self.OTF = self.OTF_form(fc_ratio=1)
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
        crop_size=self.image_size
        augmented_images = []
        for image in images:
            h, w = image.size
            pad_w = max(crop_size - w, 0)
            pad_h = max(crop_size - h, 0)
            img_pad = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(image)
            center_crop = transforms.CenterCrop(size=(crop_size, crop_size))
            imag_pad_crop = center_crop(img_pad)
            image_gray = imag_pad_crop.convert('L')
            TensorImage = transforms.ToTensor()(image_gray)

            # augmented_images += self.LR_image_generator(imag_pad_crop)
            augmented_images += self.SR_image_generator(TensorImage)
            augmented_images += self.LR_image_generator(TensorImage)
            augmented_images += self.SinusoidalPattern(TensorImage)
            # print('hello')

        return augmented_images

    def SinusoidalPattern(self, image):
        '''
        :param image:  PIL_Image that will be loaded pattern on
        :param NumPhase:  Number of phase
        :return: SinusoidalPatternImage: Image which loaded sinusoidal pattern
        '''
        resolution = 0.61 * self.EmWaveLength / self.NA
        # xx, yy, _, _ = self.GridGenerate(image=torch.rand(7, 7))
        # xx, yy, fx, fy = self.GridGenerate(image)
        SinPatternPIL_Image = []
        SIMdata_PIL_Image = []
        random_initial_direction_phase = random.random() * 2 * math.pi
        for i in range(3):
            theta = i * 2 / 3 * math.pi + random_initial_direction_phase
            SpatialFrequencyX = -self.pattern_frequency_ratio * 1 / resolution * math.sin(theta)  # 0.8倍的极限频率条纹 pattern_frequency_ratio，可调
            SpatialFrequencyY = -self.pattern_frequency_ratio * 1 / resolution * math.cos(theta)
            random_initial_phase = random.random() * 2 * math.pi
            for j in range(self.NumPhase):
                phase = j * 2 / self.NumPhase * math.pi + random_initial_phase
                SinPattern = (torch.cos(
                    phase + 2 * math.pi * (SpatialFrequencyX * self.xx + SpatialFrequencyY * self.yy)) + 1) / 2
                SIMdata_OTF_filter = self.OTF_Filter(SinPattern * image,self.OTF)
                SIMdata_OTF_filter_gaussian_noise = self.add_gaussian_noise(SIMdata_OTF_filter)
                SIMdata_OTF_filter_gaussian_noise = SIMdata_OTF_filter_gaussian_noise.float()
                SIMdata_OTF_filter_gaussian_noise_PIL = transforms.ToPILImage()(SIMdata_OTF_filter_gaussian_noise).convert('RGB')
                SIMdata_PIL_Image.append(SIMdata_OTF_filter_gaussian_noise_PIL)

                SinPattern = SinPattern.float()
                SinPattern_PIL = transforms.ToPILImage()(SinPattern).convert('RGB')
                SinPatternPIL_Image.append(SinPattern_PIL)

        return SIMdata_PIL_Image + SinPatternPIL_Image

    def GridGenerate(self, image_size):
        '''
        :param Magnification: the magnification of the Microscope
        :param PixelSize: the PixleSize of the sCMOS or CCD
        :param EmWaveLength:  emission wavelength of sample
        :param NA:  NA(numerical aperture) of the objective
        :return:
        '''
        y, x = image_size, image_size
        PixelSize = self.PixelSizeOfCCD / self.Magnification
        delta_x = PixelSize  # xy方向的空域像素间隔，单位m
        delta_y = PixelSize
        delta_fx = 1 / x / delta_x  # xy方向的频域像素间隔，单位m ^ -1
        delta_fy = 1 / y / delta_y
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
        fx = xx * delta_fx
        fy = yy * delta_fy
        xx = xx * delta_x
        yy = yy * delta_y
        return xx, yy, fx, fy

    def OTF_Filter(self, image, OTF):
        image = image.squeeze()
        if len(image.size()) == 3:
            image_PIL = transforms.ToPILImage()(image).convert('RGB')
            NumpyImage = np.asarray(image_PIL)
        else:
            NumpyImage = image.numpy()
        FFT_NumpyImage = np.fft.fft2(NumpyImage, axes=(0, 1))
        # def OTF_Filter(image):
        # image_PIL = transforms.ToPILImage()(image).convert('RGB')
        # _, _, fx, fy = self.GridGenerate(image=image_PIL)
        dim = OTF.shape

        if len(FFT_NumpyImage.shape) == 3:
            FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) * OTF.numpy().reshape(dim[0], dim[1],
                                                                                                      1)  # reshape the OTF to fullfill the broadcast mechanism
        elif len(FFT_NumpyImage.shape) == 2:
            FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) * OTF.numpy()
        else:
            raise Exception('The dimensions of input images must be 2 or 3 ')

        Filter_NumpyImage = np.fft.ifft2(np.fft.ifftshift(FilterFFT_NumpyImage, axes=(0, 1)), axes=(0, 1))
        Filter_NumpyImage = abs(Filter_NumpyImage)

        Filter_NumpyImage = Filter_NumpyImage/Filter_NumpyImage.max()*256

        filter_tensor_Image = torch.from_numpy(Filter_NumpyImage)
        return filter_tensor_Image

    def add_gaussian_noise(self, tensor_Image):  # The type of input image is PIL
        # if len(TensorImage)==3:
        #      TensorImage = TensorImage.permute(1, 2, 0) # transope for matplot
        # signal_intensity_of_image = (tensor_Image ** 2).mean()  # The mean intensity of signal
        signal_std_of_image = (tensor_Image ** 2).std() # The std intensity of signal
        noise_std_of_image = signal_std_of_image / self.SNR
        noise_of_image = torch.zeros_like(tensor_Image)
        # std_of_noise = noise_std_of_image ** (0.5)
        noise_of_image.normal_(mean=0, std=noise_std_of_image)
        image_with_noise = tensor_Image + noise_of_image
        image_with_noise = torch.where(image_with_noise < 0, torch.zeros_like(image_with_noise), image_with_noise)
        image_with_noise_normalized = image_with_noise / image_with_noise.max()
        return image_with_noise_normalized
        # return image_with_noise_normalized

    def SR_image_generator(self, TensorImage):

        OTF = self.OTF_form(fc_ratio=1.9)
        SR_image_tensor= self.OTF_Filter(TensorImage,OTF)
        SR_image_tensor_normalized = SR_image_tensor / SR_image_tensor.max()
        SR_image_tensor_normalized = SR_image_tensor_normalized.float()
        SR_image_PIL = transforms.ToPILImage()(SR_image_tensor_normalized).convert('RGB')
        return [SR_image_PIL]

    def LR_image_generator(self, TensorImage):

        OTF = self.OTF_form(fc_ratio=1)
        LR_image_tensor= self.add_gaussian_noise(self.OTF_Filter(TensorImage,OTF))
        LR_image_tensor = LR_image_tensor.float()
        LR_image_PIL = transforms.ToPILImage()(LR_image_tensor).convert('RGB')
        return [LR_image_PIL]

    def OTF_form(self, fc_ratio=1):
        f0 = fc_ratio * self.f0
        f = self.f
        OTF = torch.where(f < f0, (2 / math.pi) * (torch.acos(f / f0) - (f / f0) * (
            pow((1 - (self.f / f0) ** 2), 0.5))), torch.Tensor([0]))  # Caculate the OTF support
        # OTF = torch.where(f < f0,torch.ones_like(f),torch.zeros_like(f))
        return OTF

    def psf_form(self, OTF):

        OTF = OTF.squeeze()
        Numpy_OTF = OTF.numpy()
        psf = np.fft.ifftshift(np.fft.ifft2(Numpy_OTF, axes=(0, 1)),axes=(0, 1))
        psf = abs(psf)
        psf_Numpy = psf / psf.max()
        psf_tensor = torch.from_numpy(psf_Numpy)
        half_size_of_psf = int(psf.shape[0] / 2)
        half_row_of_psf = psf_tensor[half_size_of_psf]
        # a = half_row_of_psf < 1e-2
        id = torch.arange(0, half_row_of_psf.nelement())[half_row_of_psf.gt(1e-2)]
        psf_crop = psf_tensor[id[0]:id[-1] + 1, id[0]:id[-1] + 1]
        return psf_crop

class psf_conv_generator(nn.Module):
    def __init__(self,kernal):
        super(psf_conv_generator, self).__init__()
        self.kernal = kernal
    def forward(self, HR_image,device):
        HR_image = HR_image.squeeze()
        kernal_size = self.kernal.size()[0]
        dim_of_HR_image = len(HR_image.size())
        if dim_of_HR_image == 4:
            min_batch = HR_image.size()[0]
            channels = HR_image.size()[1]
        elif dim_of_HR_image == 3:
            channels = HR_image.size()[0]
            HR_image = HR_image.view(1,channels,HR_image.size()[1],HR_image.size()[2])
        else:
            channels = 1
            HR_image = HR_image.view(1, 1, HR_image.size()[0], HR_image.size()[1])
        out_channel = channels
        kernel = torch.FloatTensor(self.kernal).expand(out_channel, 1, kernal_size, kernal_size)
        kernel.to(device)
        # self.weight = nn.Parameter(data=kernel, requires_grad=False)
        return F.conv2d(HR_image,kernel.to(device), stride= 1, padding= int((kernal_size-1)/2),groups = out_channel)

def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)

if __name__ == '__main__':
    source_directory = 'D:\DataSet\DIV2K\subimage\getImage.jpg'
    HR_image = Image.open(source_directory)
    temp = SinusoidalPattern(probability = 1)
    OTF = temp.OTF
    psf = temp.psf_form(OTF)
    HR_image = transforms.ToTensor()(HR_image.convert('L'))
    psf_conv_instance = psf_conv_generator(psf)
    LR_image = psf_conv_instance(HR_image)
    LR_image = LR_image.squeeze()
    LR_image = LR_image/LR_image.max()
    LR_image_PIL = transforms.ToPILImage()(LR_image)
    LR_image_PIL.show()
    # psf_crop_PIL = transforms.ToPILImage()(psf_crop)
    # psf_crop_PIL.show()


