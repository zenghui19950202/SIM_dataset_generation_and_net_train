#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/29
import torch
import math
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from parameter_estimation import estimate_SIM_pattern_parameters
import torch.nn as nn
import time
import torch.optim as optim
import numpy as np
from numpy.fft import fft2
from numpy.fft import fftshift
from numpy.fft import ifft2
from numpy.fft import ifftshift
from self_supervised_learning_sr.processing_utils import winier_filter
from self_supervised_learning_sr.processing_utils import high_pass_filter
import multiprocessing
from simulation_data_generation import fuctions_for_generate_pattern as funcs

def estimate_SIM_pattern_and_parameters_of_multichannels(SIM_data):
    batch_size, input_channel, image_size, _ = SIM_data.shape
    experimental_parameters = SinusoidalPattern(probability=1, image_size=image_size)
    if experimental_parameters.upsample == True:
        SR_image_size = experimental_parameters.SR_image_size
        estimated_SIM_pattern = torch.zeros([batch_size, input_channel, SR_image_size, SR_image_size],
                                            device=SIM_data.device)
    else:
        estimated_SIM_pattern = torch.zeros_like(SIM_data)
    image_size = experimental_parameters.image_size
    estimated_SIM_pattern_parameters = torch.zeros(input_channel, 5)
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size, grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)
    for i in range(input_channel):
        one_channel_SIM_data = SIM_data[:, i, :, :].squeeze()
        estimated_spatial_frequency, estimated_modulation_factor,I0 = estimate_SIM_pattern_parameters.calculate_spatial_frequency(
            one_channel_SIM_data * one_channel_SIM_data)
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data,
                                                                          estimated_spatial_frequency)
        if abs(math.sin(estimated_phase)) > 0.1:  #
            estimated_modulation_factor = estimate_SIM_pattern_parameters.calculate_modulation_factor(one_channel_SIM_data,
                                                                            estimated_spatial_frequency,
                                                                            estimated_phase)
        else:
            rolled_one_channel_SIM_data = torch.roll(one_channel_SIM_data, [2, 0], [1, 0])
            estimated_phase_rolled = estimate_SIM_pattern_parameters.calculate_phase(rolled_one_channel_SIM_data,
                                                                              estimated_spatial_frequency)
            estimated_modulation_factor = estimate_SIM_pattern_parameters.calculate_modulation_factor(rolled_one_channel_SIM_data,
                                                                            estimated_spatial_frequency,
                                                                            estimated_phase_rolled)
        # estimated_modulation_factor = 1
        estimated_SIM_pattern_parameters[i, :] = torch.tensor(
            [*estimated_spatial_frequency, estimated_modulation_factor, torch.tensor(estimated_phase),torch.tensor(I0)])
        if experimental_parameters.upsample == True:
            estimated_SIM_pattern[:, i, :, :] = (estimated_modulation_factor * torch.cos(
                estimated_phase + 2 * math.pi * (
                        estimated_spatial_frequency[0] / 2 * xx / image_size + estimated_spatial_frequency[
                    1] / 2 * yy / image_size)) + 1) / 2
        else:
            estimated_SIM_pattern[:, i, :, :] = (estimated_modulation_factor * torch.cos(
                estimated_phase + 2 * math.pi * (
                        estimated_spatial_frequency[0] * xx / image_size + estimated_spatial_frequency[
                    1] * yy / image_size)) + 1) / 2

    estimated_SIM_pattern_parameters[:,4] = estimated_SIM_pattern_parameters[:,4] / estimated_SIM_pattern_parameters[:,4].max()
    return estimated_SIM_pattern, estimated_SIM_pattern_parameters

def estimate_SIM_pattern_and_parameters_of_multichannels_V1(SIM_data,experimental_parameters):
    batch_size, input_channel, image_size, _ = SIM_data.shape
    if experimental_parameters.upsample == True:
        SR_image_size = experimental_parameters.SR_image_size
        estimated_SIM_pattern = torch.zeros([batch_size, input_channel, SR_image_size, SR_image_size],
                                            device=SIM_data.device)
        estimated_SIM_pattern_without_m = torch.zeros([batch_size, input_channel, SR_image_size, SR_image_size],
                                            device=SIM_data.device)
    else:
        estimated_SIM_pattern = torch.zeros_like(SIM_data)
        estimated_SIM_pattern_without_m = torch.zeros_like(SIM_data)

    image_size = experimental_parameters.image_size
    estimated_SIM_pattern_parameters = torch.zeros(input_channel, 5)
    xx, yy, _, _ = experimental_parameters.GridGenerate(grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)
    for i in range(input_channel):
        one_channel_SIM_data = SIM_data[:, i, :, :].squeeze()
        start_time = time.time()
        estimated_spatial_frequency, estimated_modulation_factor,I0 = estimate_SIM_pattern_parameters.calculate_spatial_frequency(
            one_channel_SIM_data * one_channel_SIM_data)
        end_time = time.time()
        # print('the time spent on calculating the spatial frequency: %f' %(end_time-start_time))
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data,
                                                                          estimated_spatial_frequency)

        if abs(math.sin(estimated_phase)) > 0.1:  #
            m = estimate_SIM_pattern_parameters.calculate_modulation_factor(one_channel_SIM_data,
                                                                            estimated_spatial_frequency,
                                                                            estimated_phase)
        else:
            rolled_one_channel_SIM_data = torch.roll(one_channel_SIM_data, [2, 0], [1, 0])
            estimated_phase_rolled = estimate_SIM_pattern_parameters.calculate_phase(rolled_one_channel_SIM_data,
                                                                              estimated_spatial_frequency)
            m = estimate_SIM_pattern_parameters.calculate_modulation_factor(rolled_one_channel_SIM_data,
                                                                            estimated_spatial_frequency,
                                                                            estimated_phase_rolled)
        # m = 1
        estimated_SIM_pattern_parameters[i, :] = torch.tensor(
            [*estimated_spatial_frequency, m, torch.tensor(estimated_phase),torch.tensor(I0)])
        if experimental_parameters.upsample == True:
            estimated_SIM_pattern[:, i, :, :] = (m * torch.cos(
                estimated_phase + 2 * math.pi * (
                        estimated_spatial_frequency[0] / 2 * xx / image_size + estimated_spatial_frequency[
                    1] / 2 * yy / image_size)) + 1) / 2
        else:
            estimated_SIM_pattern[:, i, :, :] = (m * torch.cos(
                estimated_phase + 2 * math.pi * (
                        estimated_spatial_frequency[0] * xx / image_size + estimated_spatial_frequency[
                    1] * yy / image_size)) + 1) / 2
            estimated_SIM_pattern_without_m[:, i, :, :] =  torch.cos(estimated_phase + 2 *
                math.pi * (estimated_spatial_frequency[0] * xx / image_size + estimated_spatial_frequency[1] * yy / image_size))
    estimated_SIM_pattern_parameters[:, 4] = estimated_SIM_pattern_parameters[:, 4] / estimated_SIM_pattern_parameters[:, 4].max()
    return estimated_SIM_pattern, estimated_SIM_pattern_parameters, estimated_SIM_pattern_without_m

def estimate_SIM_pattern_and_parameters_of_TIRF_image(input_SIM_data, SIM_data, experimental_parameters):
    batch_size, channel_num, image_size, _ = SIM_data.shape
    input_channel_num = input_SIM_data.size()[1]
    estimated_SIM_pattern_parameters = torch.zeros(6, 4)
    xx, yy, _, _ = experimental_parameters.GridGenerate(grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)

    if experimental_parameters.upsample == True:
        SR_image_size = experimental_parameters.SR_image_size
        estimated_SIM_pattern = torch.zeros([batch_size, channel_num, SR_image_size, SR_image_size],
                                            device=input_SIM_data.device)
    else:
        estimated_SIM_pattern = torch.zeros_like(input_SIM_data)

    wide_field_image = torch.zeros([3, image_size, image_size])
    if channel_num == 9:
        for i in range(3):
            wide_field_image[i, :, :] = torch.mean(SIM_data[:, i * 3:(i + 1) * 3, :, :].squeeze(), dim=0)
    elif channel_num == 6:
        for i in range(3):
            wide_field_image[i, :, :] = torch.mean(SIM_data[:, i * 2:(i + 1) * 2, :, :].squeeze(), dim=0)

    for i in range(6):
        one_channel_SIM_data = input_SIM_data[:, i, :, :].squeeze()
        direction_num = i // 2
        wide_field_image_direction = wide_field_image[direction_num, :, :]
        estimated_spatial_frequency = estimate_spatial_frequency_pre_filtered_cross_correlation(wide_field_image_direction, one_channel_SIM_data)
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data, estimated_spatial_frequency)

        if abs(math.sin(estimated_phase)) > 0.1:  #
            m = estimate_SIM_pattern_parameters.calculate_modulation_factor(one_channel_SIM_data,
                                                                            estimated_spatial_frequency,
                                                                            estimated_phase)
        else:
            rolled_one_channel_SIM_data = torch.roll(one_channel_SIM_data, [2, 0], [1, 0])
            estimated_phase_rolled = estimate_SIM_pattern_parameters.calculate_phase(rolled_one_channel_SIM_data,
                                                                                     estimated_spatial_frequency)
            m = estimate_SIM_pattern_parameters.calculate_modulation_factor(rolled_one_channel_SIM_data,
                                                                            estimated_spatial_frequency,
                                                                            estimated_phase_rolled)
        # m = 1
        estimated_SIM_pattern_parameters[i, :] = torch.tensor(
            [*estimated_spatial_frequency, m, torch.tensor(estimated_phase)])
        if experimental_parameters.upsample == True:
            estimated_SIM_pattern[:, i, :, :] = (m * torch.cos(
                estimated_phase + 2 * math.pi * (
                        estimated_spatial_frequency[0] / 2 * xx / image_size + estimated_spatial_frequency[
                    1] / 2 * yy / image_size)) + 1) / 2
        else:
            estimated_SIM_pattern[:, i, :, :] = (m * torch.cos(
                estimated_phase + 2 * math.pi * (
                        estimated_spatial_frequency[0] * xx / image_size + estimated_spatial_frequency[
                    1] * yy / image_size)) + 1) / 2

    return estimated_SIM_pattern,estimated_SIM_pattern_parameters

def estimate_spatial_frequency_pre_filtered_cross_correlation(wide_field_image_direction,SIM_data):
    """@brief ： esitmate spatial frequency of raw SIM images by using pre-filtered cross correlation algorithm
      @param wide_field_image_direction: wide field image obtained by averaging  SIM raw images in the same illumination direction.
      @return spatial_frequency: spatial frequency of raw SIM images
    ."""
    # device = SIM_data.device
    SIM_data = SIM_data.squeeze()
    MSE_loss = nn.MSELoss()
    num_epochs = 50
    device = 'cpu'
    params = []
    alpha = torch.ones(1)
    alpha.requires_grad = True
    params += [{'params': alpha}]
    optimizer_alpha = optim.Adam(params, lr=0.01)

    for epoch in range(num_epochs):
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer_alpha.zero_grad()
        mse_loss = MSE_loss(torch.zeros_like(SIM_data), SIM_data - alpha * wide_field_image_direction)
        loss += mse_loss
        loss.backward()
        optimizer_alpha.step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    SIM_first_order = SIM_data - alpha.detach() * wide_field_image_direction
    cross_correlation_image = high_pass_filter(winier_filter(wide_field_image_direction)) * winier_filter(SIM_first_order)
    fft_numpy_wide_field = fftshift(fft2(high_pass_filter(winier_filter(wide_field_image_direction)).numpy(), axes=(0, 1)), axes=(0, 1))
    fft_SIM_first_order = fftshift(fft2(winier_filter(SIM_first_order).numpy(), axes=(0, 1)), axes=(0, 1))

    image = cross_correlation_image.squeeze()
    image_size = image.size()[0]
    experimental_parameters = SinusoidalPattern(probability = 1,image_size = image_size)

    image_np = image.detach().numpy()
    fft_numpy_image = fftshift(fft2(image_np, axes=(0, 1)), axes=(0, 1))
    abs_fft_np_image = np.log(1+abs(fft_numpy_image))

    if image_size % 2 == 0:
        padding_size = int(image_size / 2)
        fft_SIM_first_order_pad = np.pad(fft_SIM_first_order,
               ((padding_size, padding_size), (padding_size, padding_size)))
    else:
        padding_size = int(image_size / 2)
        fft_SIM_first_order_pad = np.pad(fft_SIM_first_order,
               ((padding_size, padding_size + 1), (padding_size, padding_size + 1)))


    # pixel_frequency = complex_cross_correlation_by_fft(high_pass_filter(SIM_data).numpy(), 1)
    pixel_frequency = complex_cross_correlation_by_fft(winier_filter(SIM_first_order).numpy(),
                                                       high_pass_filter(winier_filter(wide_field_image_direction)).numpy())
    # match_array = complex_cross_correlation(fft_SIM_first_order_pad, fft_numpy_wide_field)
    #
    # match_array_size = match_array.shape
    # x = torch.linspace(1,match_array_size[0],match_array_size[0])
    # fx,_ = torch.meshgrid(x,x)
    # mask_right_half = torch.where(fx >= match_array_size[0]//2, torch.Tensor([1]), torch.Tensor([0])).numpy()
    # filtered_match_array = match_array * mask_right_half
    # match_array_spatial = ifft2(ifftshift(match_array, axes=(0, 1)), axes=(0, 1))
    #
    #
    # peak_position_pixel = np.unravel_index(np.argmax(filtered_match_array), filtered_match_array.shape)
    # x0 = torch.tensor(peak_position_pixel)
    #
    # max_iteritive = 400
    # #TODO:怀疑这里的pattern估计有问题，明明系数已经很接近了，但重构结果还是很差，查查原因
    # peak_subpixel_location,estimated_modulation_facotr = estimate_SIM_pattern_parameters.maxmize_shift_peak_intesity(match_array_spatial, peak_position_pixel, max_iteritive)
    # frequency_peak = x0 + peak_subpixel_location
    #
    # center_position = [math.ceil(match_array_size[0] // 2),math.ceil(match_array_size[1] // 2)]
    # pixel_frequency = frequency_peak - torch.tensor(center_position)


    return torch.from_numpy(pixel_frequency)

def fine_adjust_SIM_pattern(input_SIM_raw_data, intial_estimated_pattern_params, modulation_factor, xx, yy):
    Tanh = nn.Tanh()
    delta_modulation = abs(Tanh(modulation_factor))
    # pattern_params = intial_estimated_pattern_params + delta_pattern_params
    estimated_SIM_pattern = torch.zeros_like(input_SIM_raw_data)
    channels = input_SIM_raw_data.shape[1]
    image_size = input_SIM_raw_data.shape[2]

    for i in range(channels):
        estimated_spatial_frequency_x = intial_estimated_pattern_params[i][0]
        estimated_spatial_frequency_y = intial_estimated_pattern_params[i][1]
        estimated_modulation_factor = delta_modulation[i]
        estimated_phase = intial_estimated_pattern_params[i][3]

        estimated_SIM_pattern[:, i, :, :] = (estimated_modulation_factor * torch.cos(
            estimated_phase + 2 * math.pi * (
                    estimated_spatial_frequency_x * xx / image_size + estimated_spatial_frequency_y * yy / image_size)) + 1) / 2

    return estimated_SIM_pattern

def fine_adjust_SIM_pattern_V1(estimated_SIM_pattern_without_m, modulation_factor,estimated_SIM_pattern_parameters):
    Tanh = nn.Tanh()
    modulation_factor = abs(Tanh(modulation_factor))
    image_size = estimated_SIM_pattern_without_m.size()
    modulation_factor = modulation_factor.view(1,image_size[1],1,1)
    illumination_intensity = estimated_SIM_pattern_parameters[:,4]
    illumination_intensity = illumination_intensity.view(1,image_size[1],1,1)
    estimated_SIM_pattern = illumination_intensity * (estimated_SIM_pattern_without_m * modulation_factor + 1) / 2.0
    return estimated_SIM_pattern

def calculate_pattern_frequency_ratio(estimated_pattern_parameters):
    experimental_parameters = SinusoidalPattern(probability=1)
    f_cutoff = experimental_parameters.f_cutoff
    spatial_freq = estimated_pattern_parameters[:, 0:2].squeeze()
    input_num = spatial_freq.size()[0]
    spatial_freq = 0
    for i in range(3):
        if input_num == 9:
            spatial_freq_mean = torch.mean(spatial_freq[0 + 3 * i:3 + 3 * i, :], dim=0)
        elif input_num == 5:
            if i == 0:
                spatial_freq_mean = torch.mean(spatial_freq[0:3, :], dim=0)
            else:
                spatial_freq_mean = spatial_freq[2 + i, :]
        elif input_num == 4:
            spatial_freq_mean = spatial_freq[i, :]

        spatial_freq += pow(pow(spatial_freq_mean[0], 2) + pow(spatial_freq_mean[1], 2), 1 / 2)

    spatial_freq = spatial_freq / 3
    estimated_frequency_ratio = spatial_freq / f_cutoff

    return estimated_frequency_ratio

def complex_cross_correlation(image,template):
    """@brief ：calculate complex cross correlation between input image and template
      @param image, np array. frequency spectrum image of SIM subtract wide field frequency spectrum
      @param template, np array. wide field frequency spectrum
      @return match_array: correlation array
    ."""
    global  global_image, global_template
    global_image = image
    global_template= template
    image_size = image.shape[0]
    template_size = template.shape[0]
    match_array_size = image_size-template_size + 1
    match_array = np.zeros([match_array_size,match_array_size]).astype(complex)
    # for i in range(match_array_size):
    #     for j in range(match_array_size):
    #         local = image[i:i + template_size, j:j + template_size ]
    #         normalized_local = local/np.max(abs(local))
    #         match_array[i, j] = np.sum( normalized_local * template)
    #         print(i,j)

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    index = np.linspace(0, match_array_size**2-1, match_array_size**2)

    match_array_list = pool.map(template_match, index)
    match_array = np.array(match_array_list)
    match_array_reshape = match_array.reshape(match_array_size,match_array_size)

    return abs(match_array_reshape)

def complex_cross_correlation_by_fft(SIM_image_1st,SIM_image_0th):
    """@brief ：calculate complex cross correlation between input image and template using FFT
      @param image, np array. frequency spectrum image of SIM subtract wide field frequency spectrum
      @param template, np array. wide field frequency spectrum
      @return match_array: correlation array
    ."""
    # First upsample by  a  factor  of  2  to  obtain  initial  estimate,
    # Embed Fourier data in a 2x larger array
    x_size, y_size = SIM_image_1st.shape
    x_size_large = x_size * 2
    y_size_large = y_size * 2
    CC = np.zeros([x_size_large, y_size_large],dtype = complex)
    CC[x_size - np.int(np.fix(x_size / 2)): x_size + 1 + np.int(np.fix((x_size - 1) / 2)), y_size - np.int(np.fix(y_size / 2)): y_size + 1 + np.int(np.fix((y_size - 1) / 2))] = SIM_image_1st * SIM_image_0th

    # Compute crosscorrelation and locate the peak
    CC = fftshift(fft2(CC)) # Calculate cross - correlation
    CC_peak_loc = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)

    # Obtain shift in original pixel grid from the position of the
    # crosscorrelation peak

    x_size_mid, y_size_mid = x_size_large // 2, y_size_large // 2

    x_size,y_size = CC.shape

    row_shift = CC_peak_loc[0] - x_size_mid
    col_shift = CC_peak_loc[1] - y_size_mid

    row_shift=row_shift/2
    col_shift=col_shift/2

    row_shift_sub_pixel,col_shift_sub_pixel = subpixel_register(SIM_image_1st,SIM_image_0th,row_shift,col_shift,usfac=100)

    return np.array([row_shift_sub_pixel,col_shift_sub_pixel])

def subpixel_register(SIM_image_1st,SIM_image_0th,row_shift,col_shift,usfac=100):
    x_size, y_size = SIM_image_1st.shape
    dft_center = np.int( np.fix(np.ceil(usfac * x_size) / 2) ) - 1  # Center of output array at dftshift + 1
    peak_loc_row = row_shift * usfac + dft_center
    peak_loc_col = col_shift * usfac + dft_center

    # row_shift = np.int(np.round(row_shift * usfac) / usfac )
    # col_shift = np.int(np.round(col_shift * usfac) / usfac)
    dftshift = np.int(np.fix(np.ceil(usfac * 1.5) / 2))
    # CC = np.conj(dftups(SIM_image_1st * np.conj(SIM_image_0th), usfac, dftshift-row_shift*usfac, dftshift-col_shift*usfac))

    if peak_loc_row > dft_center:
        peak_loc_row = peak_loc_row - (dft_center+1)
    else:
        peak_loc_row += peak_loc_row + (dft_center+1)

    if peak_loc_col > dft_center:
        peak_loc_col = peak_loc_col - (dft_center+1)
    else:
        peak_loc_col = peak_loc_col + (dft_center+1)

    # Matrix multiply  DFT around the  current  shift  estimate
    CC = np.conj(dftups(SIM_image_1st * np.conj(SIM_image_0th), usfac, peak_loc_row, peak_loc_col ))
    # Locate maximum and map back to original  pixel  grid
    rloc, cloc = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)
    rloc = rloc - dftshift + 1
    cloc = cloc - dftshift + 1
    row_shift_result = row_shift + rloc / usfac
    col_shift_result = col_shift + cloc / usfac

    return row_shift_result, col_shift_result

def dftups(CC_in_spatial,usfac=100,roff=0,coff=0):
    """
    :param CC:
    :param nor:
    :param noc:
    :param usfac:
    :param roff:
    :param coff:
    :return:
    """
    nr, nc = CC_in_spatial.shape
    nor = np.int(np.ceil(usfac * 1.5))
    noc = np.int(np.ceil(usfac * 1.5))
    # Compute kernels and obtain DFT by matrix products
    kernc = np.exp((-1j * 2 * math.pi ) * np.matmul(np.linspace(0,nc-1,nc).reshape(nc,1), ( np.linspace(0,noc-1,noc)- np.floor((noc-1)/2) - coff ).reshape(1,noc)/ (nc * usfac) ) )
    kernr = np.exp((-1j * 2 * math.pi ) * np.matmul( (np.linspace(0,nor-1,nor).reshape(nor,1) - np.floor((nor-1)/2) - roff)/ (nr * usfac), np.linspace(0,nr-1,nr).reshape(1,nr) ) )
    # kernc = np.exp((-1j * 2 * math.pi / (nc * usfac)) * np.matmul(ifftshift(np.linspace(0, nc - 1, nc).reshape(nc, 1))- np.floor((nc-1)/2),
    #                                                               (np.linspace(0, noc - 1, noc) - coff).reshape(1,noc)))
    # kernr = np.exp((-1j * 2 * math.pi / (nr * usfac)) * np.matmul((np.linspace(0, nor - 1, nor).reshape(nor, 1) - roff),
    #                                                               ifftshift(np.linspace(0, nr - 1, nr)).reshape(1, nr)- np.floor((nr-1)/2)))

    # kernc = np.exp((-1j * 2 * math.pi / (nc)) * np.matmul(np.linspace(0, nc - 1, nc).reshape(nc, 1), (
    #             np.linspace(0, nc - 1, nc) - np.floor((nc - 1) / 2) - 50).reshape(1, nc)))
    # kernr = np.exp((-1j * 2 * math.pi / (nr )) * np.matmul(
    #     (np.linspace(0, nr - 1, nr).reshape(nr, 1) - np.floor((nr - 1) / 2) - 50),
    #     np.linspace(0, nr - 1, nr).reshape(1, nr)))

    CC = np.matmul( np.matmul(kernr, CC_in_spatial), kernc)
    # experimental_params = funcs.SinusoidalPattern(probability=1, image_size=nr)
    # OTF = experimental_params.OTF_form(fc_ratio=2)
    # psf = np.abs(fftshift(ifft2(ifftshift(OTF))))
    # OTF_subpixel = np.matmul( np.matmul(kernr, psf), kernc)
    # CC /= np.abs(OTF_subpixel)
    return CC


def template_match(index):
    image_size = global_image.shape[0]
    template_size = global_template.shape[0]
    match_array_size = image_size - template_size + 1
    i = int(index // match_array_size)
    j = int(index % match_array_size)
    local = global_image[i:i + template_size, j:j + template_size]
    normalized_local = local / np.max(abs(local))
    match_array_value = np.sum(normalized_local * np.conj(global_template))

    return match_array_value

if __name__ == '__main__':
    pass
    # SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern_and_parameters_of_multichannels(input_SIM_raw_data)
