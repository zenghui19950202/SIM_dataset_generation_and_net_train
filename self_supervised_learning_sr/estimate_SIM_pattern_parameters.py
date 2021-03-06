#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/28

from numpy.fft import fft2
from numpy.fft import fftshift
from utils import load_configuration_parameters, common_utils
import math
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# from scipy.optimize import minimize
# from scipy.interpolate import griddata
# from torch.nn import functional as F
from scipy.interpolate import interp2d

def calculate_phase(image,pixel_frequency):
    image = image.squeeze()
    experimental_parameters = SinusoidalPattern(probability = 1)
    image_size = experimental_parameters.image_size
    image_np = image.detach().numpy()
    fft_image_np = fftshift(fft2(image_np, axes=(0, 1)), axes=(0, 1))
    # OTF = experimental_parameters.OTF
    # OTF_np  = OTF.detach().numpy()
    # fft_numpy_image * OTF_np.conj()
    experimental_parameters = SinusoidalPattern(probability=1)
    image_size = experimental_parameters.image_size
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size, grid_mode='pixel')
    image_np_times_phase_gradient = image_np * np.exp(-1j * 2 * math.pi * (pixel_frequency[0].float()/image_size * xx + pixel_frequency[1].float()/image_size * yy).numpy())
    translated_fft_image_np = fftshift(fft2(image_np_times_phase_gradient, axes=(0, 1)), axes=(0, 1))
    estimated_phase = -np.angle(sum(sum(np.conj(translated_fft_image_np) * fft_image_np)))

    # abs_translated_fft_image_np = abs(translated_fft_image_np)
    # abs_fft_translated_image_tensor = torch.from_numpy(np.log(abs_translated_fft_image_np+1))
    # common_utils.plot_single_tensor_image(abs_fft_translated_image_tensor)
    # abs_fft_image_np = abs(fft_image_np)
    # abs_fft_image_np_tensor = torch.from_numpy(np.log(abs_fft_image_np + 1))
    # common_utils.plot_single_tensor_image(abs_fft_image_np_tensor)

    return estimated_phase

def calculate_spatial_frequency (image):
    image = image.squeeze()
    experimental_parameters = SinusoidalPattern(probability = 1)
    image_np = image.detach().numpy()
    fft_numpy_image = fftshift(fft2(image_np, axes=(0, 1)), axes=(0, 1))
    abs_fft_np_image = abs(fft_numpy_image)
    f0 = 0.5 * experimental_parameters.f_cutoff
    f = experimental_parameters.f
    mask_high_freq = torch.where(f > f0, torch.Tensor([1]), torch.Tensor([0])).numpy()
    mask_right_half = torch.where(f > 0, torch.Tensor([1]), torch.Tensor([0])).numpy()
    filtered_fft_raw_SIM_imag = abs_fft_np_image * mask_high_freq * mask_right_half
    high_freq_fft_raw_SIM_imag = fft_numpy_image * mask_high_freq

    peak_position_pixel = np.unravel_index(np.argmax(filtered_fft_raw_SIM_imag), filtered_fft_raw_SIM_imag.shape)
    x0 = torch.tensor(peak_position_pixel)

    max_iteritive = 400
    peak_subpixel_location,estimated_modulation_facotr = maxmize_shift_peak_intesity(image_np, peak_position_pixel, max_iteritive)
    frequency_peak = x0 + peak_subpixel_location

    image_size = image.size()
    center_position = [math.ceil(image_size[0] / 2),math.ceil(image_size[1] / 2)]
    pixel_frequency = frequency_peak - torch.tensor(center_position)

    return pixel_frequency,estimated_modulation_facotr

# def find_frequency_peak(image_np ,experimental_parameters):
#     # half_image_size = experimental_parameters.image_size / 2
#     fft_numpy_image = fftshift(fft2(image_np, axes=(0, 1)), axes=(0, 1))
#     abs_fft_np_image = abs(fft_numpy_image)
#     f0 = 0.5 * experimental_parameters.f_cutoff
#     f = experimental_parameters.f
#     mask_high_freq = torch.where(f > f0, torch.Tensor([1]), torch.Tensor([0])).numpy()
#     mask_right_half = torch.where(f > 0, torch.Tensor([1]), torch.Tensor([0])).numpy()
#     filtered_fft_raw_SIM_imag = abs_fft_np_image * mask_high_freq * mask_right_half
#     high_freq_fft_raw_SIM_imag = fft_numpy_image * mask_high_freq
#
#     peak_position_pixel = np.unravel_index(np.argmax(filtered_fft_raw_SIM_imag), filtered_fft_raw_SIM_imag.shape)
#     x0 = torch.tensor(peak_position_pixel)
#
#     max_iteritive = 400
#     peak_subpixel_location = maxmize_shift_peak_intesity(image_np, peak_position_pixel, max_iteritive)
#     frequency_peak = x0+ peak_subpixel_location
#     # result = minimize(fun=shift_correlation, x0=x0,args=(image_np,high_freq_fft_raw_SIM_imag),bounds=[(x0-0.5).numpy(),(x0+0.5).numpy()])
#     # frequency_peak = result['x']
#     # print(result.success)
# # TODO:不太能用，可能没梯度，需要更改函数，
#     # interp_scope = 51
#     # xx, yy, _, _ = experimental_parameters.GridGenerate(interp_scope, grid_mode='pixel')
#     # xx = xx + peak_position[0]
#     # yy = yy + peak_position[1]
#     # interp_values = abs_fft_np_SIM_image[peak_position[0]-int((interp_scope-1)/2) :peak_position[0] + int((interp_scope-1)/2)+1,
#     #                                      peak_position[1]-int((interp_scope-1)/2) :peak_position[1] + int((interp_scope-1)/2)+1]
#     # normalized_interp_values = -np.log(1 + interp_values)
#     # f1 = interp2d(xx, yy,normalized_interp_values , kind='cubic')
#     # result = minimize(fun=minus_interp, x0=peak_position,args=(f1), method='TNC',options={'maxiter': 400})
#     # frequency_peak = result['x']
#
#     # xx, yy, _, _ = experimental_parameters.GridGenerate(256, grid_mode='pixel')
#     # location = np.dstack((xx,yy)).reshape(-1,2)
#     # values = abs_fft_np_SIM_image.reshape(-1,1)
#     # grid_num = 200
#     # grid_xx, grid_yy = np.meshgrid(np.arange(-grid_num / 2, grid_num / 2, 1),
#     #                         np.arange(-grid_num / 2, grid_num / 2, 1))
#     # grid_xx = grid_xx / grid_num *2 + peak_position[0]
#     # grid_yy = grid_yy / grid_num *2 + peak_position[1]
#     # betagriddata = griddata(np.array([xx.numpy().ravel(), yy.numpy().ravel()]).T, values.ravel(),
#     #                                np.array([grid_xx.ravel(), grid_yy.ravel()]).T)
#     # grid_z2 = griddata(location, values, (grid_xx, grid_yy), method='linear')
#     # print(grid_z2[0][1])
#
#     return frequency_peak

def shift_freq_domain_peak_intensity(subpixel_shift,image_np,peak_position_pixel):
    experimental_parameters = SinusoidalPattern(probability=1)
    image_size = experimental_parameters.image_size
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size, grid_mode='pixel')
    image_np_times_phase_gradient = image_np * np.exp(-1j * 2 * math.pi * (subpixel_shift[0]/image_size * xx + subpixel_shift[1]/image_size * yy).numpy())
    translated_fft_image_np = fftshift(fft2(image_np_times_phase_gradient, axes=(0, 1)), axes=(0, 1))
    normalized_translated_fft_image_np = np.log(abs(translated_fft_image_np) + 1)
    normalized_translated_fft_image_np = abs(translated_fft_image_np)
    peak_intesity = normalized_translated_fft_image_np[peak_position_pixel[0],peak_position_pixel[1]]
    # np.unravel_index(np.argmax(normalized_translated_fft_image_np), normalized_translated_fft_image_np.shape)
    #
    # abs_fft_translated_image_tensor = torch.from_numpy(normalized_translated_fft_image_np)
    # common_utils.plot_single_tensor_image(abs_fft_translated_image_tensor)
    # print(subpixel_shift,peak_intesity)

    return peak_intesity

def maxmize_shift_peak_intesity(image_np,peak_position_pixel,max_iteritive = 400):
    step = 1e-1   #沿方向搜索计算时的步距，这里的单位是像素而不是空间频率
    tolerance = 1e-3   #计算结果的最高精度，比给出结果的有效位数高一个小数位
    # subpixel_location = np.zeros(2,dtype = float)
    subpixel_location = torch.zeros(2, dtype=float)
    max_peak_intensity = shift_freq_domain_peak_intensity(subpixel_location, image_np, peak_position_pixel)
    epoch = 0
    while step > tolerance and epoch < max_iteritive :
        # subpixel_location_temp = subpixel_location + np.asarray([tolerance,0], dtype = float)
        subpixel_location_temp = subpixel_location + torch.tensor([tolerance,0], dtype = float)
        temp_peak_intensity = shift_freq_domain_peak_intensity(subpixel_location_temp, image_np, peak_position_pixel)
        partial_derivative_x = (temp_peak_intensity-max_peak_intensity)/tolerance

        subpixel_location_temp = subpixel_location +  torch.tensor([0,tolerance], dtype = float)
        temp_peak_intensity = shift_freq_domain_peak_intensity(subpixel_location_temp, image_np, peak_position_pixel)
        partial_derivative_y = (temp_peak_intensity-max_peak_intensity)/tolerance
        partial_derivative_xy = torch.tensor([partial_derivative_x, partial_derivative_y])
        grad_xy = partial_derivative_xy/torch.norm(partial_derivative_xy)

        subpixel_location_temp = grad_xy * step + subpixel_location
        temp_peak_intensity = shift_freq_domain_peak_intensity(subpixel_location_temp, image_np, peak_position_pixel)

        if temp_peak_intensity > max_peak_intensity:
            subpixel_location = subpixel_location_temp
            max_peak_intensity = temp_peak_intensity
        else:
            step = step * 0.5
        # print(subpixel_location,temp_peak_intensity,max_peak_intensity)
        epoch = epoch + 1

    experimental_parameters = SinusoidalPattern(probability=1)
    image_size = experimental_parameters.image_size
    f0 = experimental_parameters.f_cutoff
    delta_fx = experimental_parameters.delta_fx
    f = torch.norm(subpixel_location + torch.tensor(peak_position_pixel)-torch.ceil(torch.tensor(image_size/2))) * delta_fx
    OTF_attenuation =  (2 / math.pi) * (torch.acos(f / f0) - (f / f0) * (pow((1 - (f/ f0) ** 2), 0.5)))
    abs_fft_image_np = abs(fftshift(fft2(image_np, axes=(0, 1)), axes=(0, 1)))
    estimated_modulation_facotr = max_peak_intensity / OTF_attenuation / abs_fft_image_np.max()

    return subpixel_location , estimated_modulation_facotr

def minus_interp(x,f1):
    return f1(x[0],x[1])

# def shift_correlation(subpixel_shift,image_np,high_freq_fft_raw_SIM_imag):
#
#     experimental_parameters = SinusoidalPattern(probability=1)
#     image_size = experimental_parameters.image_size
#     xx, yy, _, _ = experimental_parameters.GridGenerate(image_size, grid_mode='pixel')
#     image_np_times_phase_gradient = image_np * np.exp(-1j * 2 * math.pi * (subpixel_shift[0]/image_size * xx + subpixel_shift[1]/image_size * yy).numpy())
#     translated_fft_image_np = fftshift(fft2(image_np_times_phase_gradient, axes=(0, 1)), axes=(0, 1))
#     shift_correlation = -sum(sum(abs(high_freq_fft_raw_SIM_imag * translated_fft_image_np)))/sum(sum(abs(high_freq_fft_raw_SIM_imag * high_freq_fft_raw_SIM_imag)))
#
#     # normalized_translated_fft_image_np = np.log(abs(translated_fft_image_np) + 1)
#     #
#     # abs_fft_translated_image_tensor = torch.from_numpy(normalized_translated_fft_image_np)
#     # common_utils.plot_single_tensor_image(abs_fft_translated_image_tensor)
#     print(subpixel_shift,shift_correlation)
#     #
#     # normalized_translated_fft_image_np = np.log(abs(high_freq_fft_raw_SIM_imag) + 1)
#     # abs_fft_translated_image_tensor = torch.from_numpy(normalized_translated_fft_image_np)
#     # common_utils.plot_single_tensor_image(abs_fft_translated_image_tensor)
#     return shift_correlation


if __name__ == '__main__':
    # print('hello')
    estimated_phase = []
    for i in range(3):
        image_PIL = Image.open('/home/common/Zenghui/test_for_self_9_frames_supervised_SR_net/cell/SIMdata_SR_train/12084-bc404fdc-b0dd-453e-ba39-7c38c9fee270_Speckle_SIM_data('+str(i+1+3)+')_.png')
        image_PIL_gray = image_PIL.convert('L')
        TensorImage = transforms.ToTensor()(image_PIL_gray)
        # TensorImage = transforms.ToTensor()(image_PIL)
        pixel_frequency,estimated_modulation_facotr = calculate_spatial_frequency(TensorImage * TensorImage)
        estimated_phase += [calculate_phase(TensorImage,pixel_frequency)]
        print(pixel_frequency,estimated_phase[i],estimated_modulation_facotr)
    print((estimated_phase[2] - estimated_phase[1])/math.pi, (estimated_phase[1] - estimated_phase[0])/math.pi)
    # estimated_sinusoidal_pattern = estimate_SIM_pattern(pixel_frequency, phase = estimated_phase)
    # common_utils.plot_single_tensor_image(estimated_sinusoidal_pattern)

 # TODO: 注意62 行    frequency_peak = x0 + peak_subpixel_location 究竟是加还是减。。。