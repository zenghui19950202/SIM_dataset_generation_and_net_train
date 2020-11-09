#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/12
import torch
import math
from PIL import Image
from torchvision import transforms
from simulation_data_generation.generate_hologram_diffraction import hologram
from utils import *
import torch.nn as nn


def positive_propagate(SR_image, SIM_pattern, psf_conv,down_sample = False):
    AvgPooling = nn.AvgPool2d(kernel_size= 2)
    if down_sample == False:
        SIM_raw_data_estimated = psf_conv(SR_image * SIM_pattern)
    else:
        SIM_raw_data_estimated = AvgPooling(psf_conv(SR_image * SIM_pattern))
    return SIM_raw_data_estimated

def low_pass_filter(image,filter):
    image_complex = torch.stack([image, torch.zeros_like(image)], 2)
    image_complex_fft = torch_2d_fftshift(torch.fft(image_complex,2))
    image_complex_fft_filtered = image_complex_fft * filter.unsqueeze(2)
    image_complex_filtered = torch.ifft(torch_2d_ifftshift(image_complex_fft_filtered),2)
    return complex_stack_to_intensity(image_complex_filtered)

def complex_stack_to_intensity(image_complex_stack):
    if not image_complex_stack.dim() == 3:
        raise Exception('input dim is error')
    return pow( pow(image_complex_stack[:,:,0],2)+ pow(image_complex_stack[:,:,1],2), 1/2 )

def winier_deconvolution(image,OTF):
    image = image.squeeze()
    real_part_image = image
    imaginary_part_image = torch.zeros_like(image)
    image_complex = torch.stack([real_part_image, imaginary_part_image], 2)
    fft_image_complex = torch.fft(image_complex, 2)
    fft_image_complex_shifted = torch_2d_fftshift(fft_image_complex)
    fft_image_complex_shifted_winier = fft_image_complex_shifted * OTF.unsqueeze(2)/ (OTF * OTF + torch.tensor([0.04]).type_as(image)).unsqueeze(2)
    fft_image_complex_winier = torch_2d_ifftshift(fft_image_complex_shifted_winier)
    image_complex_winier = torch.ifft(fft_image_complex_winier,2)
    image_winier = pow ( pow(image_complex_winier[:,:,0],2) + pow(image_complex_winier[:,:,1],2) , 1/2)
    return image_winier

def complex_times(a,b):
    real_part_result = a[:,:,0] * b[:,:,0] - a[:,:,1] * b[:,:,1]
    imaginary_part_result = a[:,:,0] * b[:,:,1] + a[:,:,1] * b[:,:,0]
    return torch.stack([real_part_result,imaginary_part_result],2)

def complex_divide(a,b):
    b_temp = torch.zeros_like(b)
    b_temp[:,:,0] = b[:,:,0]
    b_temp[:,:,1] = b[:,:,1] * -1
    numerator  = complex_times(a,b_temp)
    denominator = pow(b[:,:,0],2) + pow(b[:,:,1],2)
    if denominator.dim() == 2:
        denominator = denominator.unsqueeze(2)
    result = numerator / denominator
    return result

def torch_2d_fftshift(image):
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

def torch_2d_ifftshift(image):
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

def expjk_to_cosk_sink_stack(k):
    if k.dim() == 2:
        result = torch.stack([torch.cos(k), torch.sin(k)], 2)
    elif k.dim() == 1:
        result = torch.stack([torch.cos(k).unsqueeze(0), torch.sin(k).unsqueeze(0)], 2)
    else:
        raise Exception('The dim of input k must be 1 or 2')
    return result

def intensity_to_complex_stack(k):
    return torch.stack([k.unsqueeze(0), torch.zeros_like(k).unsqueeze(0)], 2)

def cosk_sink_stack_to_phasek(cosk_sink_stack):

    phasek = torch.atan2(cosk_sink_stack[:,:,1],cosk_sink_stack[:,:,0])

    return phasek

def fresnel_propagate(phase_image , d , xx0, yy0 ,xx1 ,yy1, lamda,k):
    phase_image = phase_image.squeeze()
    size_x = phase_image.size()[0]
    size_y = phase_image.size()[1]

    phase_image_complex_amplitude = expjk_to_cosk_sink_stack(phase_image)

    F0_temp = complex_divide(expjk_to_cosk_sink_stack(k * d), expjk_to_cosk_sink_stack(lamda * d))
    F0 = complex_times(F0_temp, expjk_to_cosk_sink_stack(k / 2 / d * (xx1 * xx1 + yy1 * yy1)))

    F = expjk_to_cosk_sink_stack(k / 2 / d * (xx0 * xx0 + yy0 * yy0))
    phase_image_complex_amplitude = complex_times(phase_image_complex_amplitude, F)
    # phase_image_complex_amplitude = self.complex_times(rectangle, F)

    Ff = torch_2d_fftshift(torch.fft(phase_image_complex_amplitude, 2))
    Fuf = complex_times(F0, Ff)
    I = pow(Fuf[:, :, 0], 2) + pow(Fuf[:, :, 1], 2)

    return Fuf


def fresnel_reverse_propagate(phase_image_diffraction, d, xx0, yy0, xx1, yy1, lamda,k):
    phase_image_diffraction = phase_image_diffraction.squeeze()
    size_x = phase_image_diffraction.size()[0]
    size_y = phase_image_diffraction.size()[1]

    # phase_image_diffraction_complex_amplitude = expjk_to_cosk_sink_stack(phase_image_diffraction)

    F_reverse = expjk_to_cosk_sink_stack(k / 2 / d * (xx0 * xx0 + yy0 * yy0))
    F2 = torch.ifft(torch_2d_ifftshift(complex_times(F_reverse, phase_image_diffraction)), 2)
    F0_temp = complex_divide(expjk_to_cosk_sink_stack(k * d), expjk_to_cosk_sink_stack(lamda * d))
    F0_reverse = complex_times(F0_temp, expjk_to_cosk_sink_stack(k / 2 / d * (xx1 * xx1 + yy1 * yy1)))
    phase_image_recon = complex_times(F2, F0_reverse)
    I = pow(phase_image_recon[:, :, 0], 2) + pow(phase_image_recon[:, :, 1], 2)

    return phase_image_recon

if __name__ == '__main__':
    image_PIL = Image.open(
        '/data/zh/self_supervised_learning_SR/test_for_self_9_frames_supervised_SR_net/test/train/verifica_board.png')
    image_PIL_gray = image_PIL.convert('L')
    TensorImage = transforms.ToTensor()(image_PIL_gray)


    experimental_params = hologram(probability=1)
    data_num = experimental_params.data_num
    device = torch.device('cpu')

    k = torch.tensor([experimental_params.wave_num], dtype=torch.float32).to(device)
    distance = torch.tensor([experimental_params.distance], dtype=torch.float32).to(device)
    lamda = torch.tensor([experimental_params.WaveLength], dtype=torch.float32).to(device)
    xx0, xx1, yy0, yy1 = experimental_params.xx0.to(device), experimental_params.xx1.to(
        device), experimental_params.yy0.to(device), experimental_params.yy1.to(device)

    d = distance * 3
    TensorImage1 = TensorImage[0,0:256,0:256].squeeze()
    TensorImage1 = TensorImage1.to(device)
    diffraction_phase_image = fresnel_propagate(TensorImage1,d, xx0, yy0, xx1, yy1,lamda,k)
    reconstructed_phase_image = fresnel_reverse_propagate(diffraction_phase_image,-1*d, xx0, yy0, xx1, yy1,lamda,k)
    phase_image_result = cosk_sink_stack_to_phasek(reconstructed_phase_image)

    estimated_diffraction_hologram_intensity = pow(diffraction_phase_image[:,:,0],2) + pow(diffraction_phase_image[:,:,1],2)

    common_utils.plot_single_tensor_image(TensorImage1)
    common_utils.plot_single_tensor_image(estimated_diffraction_hologram_intensity)
    common_utils.plot_single_tensor_image(phase_image_result)

