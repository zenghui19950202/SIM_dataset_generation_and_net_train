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
from self_supervised_learning_sr import forward_model
from utils import load_SIM_paras_from_txt
from utils import common_utils


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
    xx, yy, _, _ = experimental_parameters.GridGenerate(grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)
    for i in range(input_channel):
        one_channel_SIM_data = SIM_data[:, i, :, :].squeeze()
        estimated_spatial_frequency, estimated_modulation_factor, I0 = estimate_SIM_pattern_parameters.calculate_spatial_frequency(
            one_channel_SIM_data * one_channel_SIM_data)
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data,
                                                                          estimated_spatial_frequency)
        if abs(math.sin(estimated_phase)) > 0.1:  #
            estimated_modulation_factor = estimate_SIM_pattern_parameters.calculate_modulation_factor(
                one_channel_SIM_data,
                estimated_spatial_frequency,
                estimated_phase)
        else:
            rolled_one_channel_SIM_data = torch.roll(one_channel_SIM_data, [2, 0], [1, 0])
            estimated_phase_rolled = estimate_SIM_pattern_parameters.calculate_phase(rolled_one_channel_SIM_data,
                                                                                     estimated_spatial_frequency)
            estimated_modulation_factor = estimate_SIM_pattern_parameters.calculate_modulation_factor(
                rolled_one_channel_SIM_data,
                estimated_spatial_frequency,
                estimated_phase_rolled)
        # estimated_modulation_factor = 1
        estimated_SIM_pattern_parameters[i, :] = torch.tensor(
            [*estimated_spatial_frequency, estimated_modulation_factor, torch.tensor(estimated_phase),
             torch.tensor(I0)])
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

    estimated_SIM_pattern_parameters[:, 4] = estimated_SIM_pattern_parameters[:, 4] / estimated_SIM_pattern_parameters[
                                                                                      :, 4].max()
    return estimated_SIM_pattern, estimated_SIM_pattern_parameters


def estimate_SIM_pattern_and_parameters_of_multichannels_V1(SIM_data, experimental_parameters):
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
        estimated_spatial_frequency, estimated_modulation_factor, I0 = estimate_SIM_pattern_parameters.calculate_spatial_frequency(
              one_channel_SIM_data)
        end_time = time.time()
        # print('the time spent on calculating the spatial frequency: %f' %(end_time-start_time))
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data,
                                                                          estimated_spatial_frequency)

        m = estimate_SIM_pattern_parameters.calculate_modulation_factor_V1(one_channel_SIM_data,
                                                                           estimated_spatial_frequency,
                                                                           estimated_phase)
        estimated_SIM_pattern_parameters[i, :] = torch.tensor(
            [*estimated_spatial_frequency, m, torch.tensor(estimated_phase), torch.tensor(I0)])
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
            estimated_SIM_pattern_without_m[:, i, :, :] = torch.cos(estimated_phase + 2 *
                                                                    math.pi * (estimated_spatial_frequency[
                                                                                   0] * xx / image_size +
                                                                               estimated_spatial_frequency[
                                                                                   1] * yy / image_size))
    estimated_SIM_pattern_parameters[:, 4] = estimated_SIM_pattern_parameters[:, 4] / estimated_SIM_pattern_parameters[
                                                                                      :, 4].max()
    return estimated_SIM_pattern, estimated_SIM_pattern_parameters, estimated_SIM_pattern_without_m


def estimate_SIM_pattern_and_parameters_of_TIRF_image(input_SIM_data, SIM_data, experimental_parameters):
    """
    :param input_SIM_data: input raw SIM for reconstruction
    :param SIM_data: All 9-frame raw SIM images for calculating polarization
    :param experimental_parameters: experimental parameters such as: wavelength, pixel size, etc.
    :return: estimated_SIM_pattern
    """
    batch_size, channel_num, image_size, _ = SIM_data.shape
    _, input_channel_num, _, _ = input_SIM_data.shape
    input_channel_num = input_SIM_data.size()[1]
    estimated_SIM_pattern_parameters = torch.zeros(input_channel_num, 4)
    xx, yy, _, _ = experimental_parameters.GridGenerate(grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)

    if experimental_parameters.upsample == True:
        SR_image_size = experimental_parameters.SR_image_size
        estimated_SIM_pattern = torch.zeros([batch_size, input_channel_num, SR_image_size, SR_image_size],
                                            device=input_SIM_data.device)
    else:
        estimated_SIM_pattern = torch.zeros_like(input_SIM_data)

    wide_field_image = torch.zeros([input_channel_num, image_size, image_size])
    "uncomment next line to use the real value of SIM illumination parameters"
    real_value_SIM_params = load_SIM_paras_from_txt.load_params()
    if input_channel_num == 9:
        for i in range(input_channel_num):
            count_num = i // 3
            wide_field_image[i, :, :] = torch.mean(SIM_data[:, count_num * 3:(count_num + 1) * 3, :, :].squeeze(),
                                                   dim=0)
    elif input_channel_num == 6 and channel_num == 6:
        for i in range(input_channel_num):
            count_num = i // 2
            wide_field_image[i, :, :] = torch.mean(SIM_data[:, count_num * 2:(count_num + 1) * 2, :, :].squeeze(),
                                                   dim=0)
    elif input_channel_num == 6 and channel_num == 9:
        real_value_SIM_params = torch.cat(
            (real_value_SIM_params[0:2, :], real_value_SIM_params[3:5, :], real_value_SIM_params[6:8, :]), 0)
        for i in range(input_channel_num):
            count_num = i // 2
            wide_field_image[i, :, :] = torch.mean(SIM_data[:, count_num * 3:(count_num + 1) * 3, :, :].squeeze(),
                                                   dim=0)

    elif input_channel_num == 7:
        wide_field_image[0, :, :] = wide_field_image[1, :, :] = wide_field_image[2, :, :] = torch.mean(
            SIM_data[:, 0:3, :, :].squeeze(), dim=0)
        wide_field_image[3, :, :] = wide_field_image[4, :, :] = torch.mean(SIM_data[:, 3:6, :, :].squeeze(), dim=0)
        wide_field_image[5, :, :] = wide_field_image[6, :, :] = torch.mean(SIM_data[:, 6:9, :, :].squeeze(), dim=0)
    elif input_channel_num == 3:
        real_value_SIM_params = torch.stack(
            (real_value_SIM_params[0, :], real_value_SIM_params[3, :], real_value_SIM_params[6, :]), 0)
    "First loop to extract paramters of illumination pattern"
    for i in range(input_channel_num):
        one_channel_SIM_data = input_SIM_data[:, i, :, :].squeeze()
        direction_num = i // 2

        wide_field_image_direction = wide_field_image[i, :, :]
        estimated_spatial_frequency, m = estimate_spatial_frequency_pre_filtered_cross_correlation(
            wide_field_image_direction, one_channel_SIM_data)
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data,
                                                                          estimated_spatial_frequency)

        estimated_spatial_frequency,m,estimated_phase = real_value_SIM_params[i,0:2], real_value_SIM_params[i,2], real_value_SIM_params[i,3]
        # m = 0.5
        estimated_SIM_pattern_parameters[i, :] = torch.tensor(
            [*estimated_spatial_frequency, m, estimated_phase])

        # if i % 2 == 1:
        #     m = calculate_modulation(input_SIM_data[:, direction_num * 2:(direction_num + 1) * 2],
        #                              wide_field_image_direction, estimated_spatial_frequency, estimated_SIM_pattern_parameters[i-1:i+1,3])
        #
        #     estimated_SIM_pattern_parameters[i-1:i+1,2] = torch.tensor([m])

    "Second loop to generate the estimated illumination pattern"
    for i in range(input_channel_num):
        spatial_frequency = estimated_SIM_pattern_parameters[i, 0:2]
        m = estimated_SIM_pattern_parameters[i, 2]
        phase = estimated_SIM_pattern_parameters[i, 3]
        if experimental_parameters.upsample == True:
            estimated_SIM_pattern[:, i, :, :] = (m * torch.cos(
                phase + 2 * math.pi * (
                        spatial_frequency[0] / 2 * xx / image_size + spatial_frequency[
                    1] / 2 * yy / image_size)) + 1)
        else:
            estimated_SIM_pattern[:, i, :, :] = (m * torch.cos(
                phase + 2 * math.pi * (
                        spatial_frequency[0] * xx / image_size + spatial_frequency[
                    1] * yy / image_size)) + 1)

    # blind_region = estimate_blind_region(estimated_SIM_pattern,estimated_SIM_pattern_parameters)

    return estimated_SIM_pattern, estimated_SIM_pattern_parameters

def estimate_blind_region(estimated_SIM_pattern,estimated_SIM_pattern_parameters):
    """
    generate blind region of structured illumination for researching the effect
    :param estimated_SIM_pattern:
    :return: blind region image
    """
    channel_num  = estimated_SIM_pattern.size()[1]
    beta = 0.5
    blind_region = torch.zeros(estimated_SIM_pattern.size()[2],estimated_SIM_pattern.size()[3])
    for i in range(channel_num):
        m = estimated_SIM_pattern_parameters[i, 2]
        temp_SIM_pattern = estimated_SIM_pattern[:, i, :, :].squeeze()
        white_region_index = torch.where(temp_SIM_pattern > (2 * m * beta + temp_SIM_pattern.min()))
        blind_region[white_region_index] = 1
        common_utils.plot_single_tensor_image(blind_region)


    return blind_region


def estimate_spatial_frequency_pre_filtered_cross_correlation(wide_field_image_direction, SIM_data):
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

        # print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))

    SIM_first_order = SIM_data - alpha.detach() * wide_field_image_direction
    cross_correlation_image = high_pass_filter(winier_filter(wide_field_image_direction)) * winier_filter(
        SIM_first_order)

    # complex_cross_correlation_by_fft函数 局部高分辨频谱寻峰，貌似精度不够
    # pixel_frequency = complex_cross_correlation_by_fft(SIM_data*SIM_data.numpy(), 1)
    # pixel_frequency = complex_cross_correlation_by_fft(winier_filter(SIM_first_order).numpy(),
    #                                                    high_pass_filter(winier_filter(wide_field_image_direction)).numpy())
    SIM_first_order_phase_only = phase_only_frequency_spectrum_est(winier_filter(SIM_first_order))
    wide_field_image_direction_phase_only = phase_only_frequency_spectrum_est(winier_filter(wide_field_image_direction))

    # CC_spatial = winier_filter(SIM_first_order).numpy() * high_pass_filter(winier_filter(wide_field_image_direction)).numpy()
    CC_spatial = SIM_first_order_phase_only * wide_field_image_direction_phase_only
    # CC_spatial = high_pass_filter(SIM_data * SIM_data)
    CC_spatial_size = CC_spatial.shape
    x = torch.linspace(1, CC_spatial_size[0], CC_spatial_size[0])
    fx, _ = torch.meshgrid(x, x)
    CC = fftshift(fft2(CC_spatial))
    mask_right_half = torch.where(fx >= CC_spatial_size[0] // 2, torch.Tensor([1]), torch.Tensor([0])).numpy()
    filtered_CC = CC * mask_right_half
    peak_position_pixel = np.unravel_index(np.argmax(abs(filtered_CC)), filtered_CC.shape)
    x0 = torch.tensor(peak_position_pixel)

    max_iteritive = 400

    peak_subpixel_location, estimated_modulation_facotr = estimate_SIM_pattern_parameters.maxmize_shift_peak_intesity(
        CC_spatial, peak_position_pixel, max_iteritive)
    frequency_peak = x0 + peak_subpixel_location

    center_position = [math.ceil(CC_spatial_size[0] // 2), math.ceil(CC_spatial_size[1] // 2)]
    pixel_frequency = frequency_peak - torch.tensor(center_position)

    " 迭代找最大值"
    # pixel_frequency = x0 - torch.tensor(center_position)
    # pixel_frequency = subpixel_cc_parameter_estimation(SIM_first_order_phase_only,wide_field_image_direction_phase_only,pixel_frequency )

    if type(pixel_frequency) is np.ndarray:
        pixel_frequency = torch.from_numpy(pixel_frequency)

    m = calculate_modulation_factor_overlap_region(SIM_first_order, wide_field_image_direction, pixel_frequency)
    return pixel_frequency, m


def subpixel_cc_parameter_estimation(SIM_1st_phase_only, WF_phase_only, peak_position_pixel):
    """
    using symmetry cross correlation to estimate illumination parameter in subpixel preicsion
    :param SIM_1st_phase_only:
    :param wide_field_phase_only:
    :param peak_position_pixel:
    :return:
    """
    peak_position_pixel = peak_position_pixel.float()
    image_size = SIM_1st_phase_only.shape[0]
    experimental_parameters = SinusoidalPattern(probability=1, image_size=image_size)
    xx, yy, _, _ = experimental_parameters.GridGenerate(grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)
    CTF = experimental_parameters.CTF
    SIM_1st_phase_only_fft = fftshift(fft2(SIM_1st_phase_only, axes=(0, 1)), axes=(0, 1))

    x = peak_position_pixel[0]
    y = peak_position_pixel[1]
    lr = 0.1
    max_iter = 500

    peak_position_pixel[0], peak_position_pixel[1] = gradient_descent(SIM_1st_phase_only_fft, WF_phase_only, x, y, xx,
                                                                      yy, CC_value_cal, lr, max_iter, CTF)

    return peak_position_pixel


def CC_value_cal(SIM_1st_fft, WF, x, y, xx, yy, CTF):
    image_size = SIM_1st_fft.shape[0]
    sin_pattern = 2 * torch.exp(1j *
                                2 * math.pi * (x * xx / image_size + y * yy / image_size)).numpy()
    WF_fft_shift = fftshift(fft2(WF * sin_pattern, axes=(0, 1)), axes=(0, 1))
    overlao_zone = fftshift((fft2(ifft2(ifftshift(CTF)) * sin_pattern))) * CTF.numpy()
    CC_value = np.sum(abs(SIM_1st_fft * WF_fft_shift * overlao_zone))

    return CC_value


def gradient_descent(SIM_1st_fft, WF_complex, x, y, xx, yy, CC_value_cal, lr, max_iter, CTF):
    tolerance = 1e-3  # 计算结果的最高精度，比给出结果的有效位数高一个小数位
    max_CC_value = CC_value_cal(SIM_1st_fft, WF_complex, x, y, xx, yy, CTF)
    for i in range(max_iter):
        # 计算出前向数值
        x1 = x + tolerance
        temp_CC_value = CC_value_cal(SIM_1st_fft, WF_complex, x1, y, xx, yy, CTF)
        grad_x = (temp_CC_value - max_CC_value) / tolerance

        y1 = y + tolerance
        temp_CC_value = CC_value_cal(SIM_1st_fft, WF_complex, x, y1, xx, yy, CTF)
        grad_y = (temp_CC_value - max_CC_value) / tolerance
        grad_xy = np.sqrt(grad_x ** 2 + grad_y ** 2)

        x2 = x + lr * grad_x / grad_xy
        y2 = y + lr * grad_y / grad_xy
        temp_CC_value = CC_value_cal(SIM_1st_fft, WF_complex, x2, y2, xx, yy, CTF)

        if temp_CC_value > max_CC_value:
            x = x2
            y = y2
            max_CC_value = temp_CC_value
        else:
            lr = lr * 0.5

    return x, y


def phase_only_frequency_spectrum_est(image):
    """
    To normalize the frequency spectrum of image and make it only contain phase information
    :param image: input numpy image
    :return: The phase-only frequency spectrum version of image
    """
    FFT_image = fft2(image)
    epsilon = 1e-4
    phase_only_FFT_image = FFT_image / (abs(FFT_image) + epsilon)
    phase_only_FFT_image_in_spatial_domain = ifft2(phase_only_FFT_image)

    return phase_only_FFT_image_in_spatial_domain


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


def fine_adjust_SIM_pattern_V1(estimated_SIM_pattern_without_m, modulation_factor, estimated_SIM_pattern_parameters):
    Tanh = nn.Tanh()
    modulation_factor = abs(Tanh(modulation_factor))
    image_size = estimated_SIM_pattern_without_m.size()
    modulation_factor = modulation_factor.view(1, image_size[1], 1, 1)
    illumination_intensity = estimated_SIM_pattern_parameters[:, 4]
    illumination_intensity = illumination_intensity.view(1, image_size[1], 1, 1)
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


def complex_cross_correlation(image, template):
    """@brief ：calculate complex cross correlation between input image and template
      @param image, np array. frequency spectrum image of SIM subtract wide field frequency spectrum
      @param template, np array. wide field frequency spectrum
      @return match_array: correlation array
    ."""
    global global_image, global_template
    global_image = image
    global_template = template
    image_size = image.shape[0]
    template_size = template.shape[0]
    match_array_size = image_size - template_size + 1
    match_array = np.zeros([match_array_size, match_array_size]).astype(complex)
    # for i in range(match_array_size):
    #     for j in range(match_array_size):
    #         local = image[i:i + template_size, j:j + template_size ]
    #         normalized_local = local/np.max(abs(local))
    #         match_array[i, j] = np.sum( normalized_local * template)
    #         print(i,j)

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    index = np.linspace(0, match_array_size ** 2 - 1, match_array_size ** 2)

    match_array_list = pool.map(template_match, index)
    match_array = np.array(match_array_list)
    match_array_reshape = match_array.reshape(match_array_size, match_array_size)

    return abs(match_array_reshape)


def complex_cross_correlation_by_fft(SIM_image_1st, SIM_image_0th):
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
    CC = np.zeros([x_size_large, y_size_large], dtype=complex)
    CC[x_size - np.int(np.fix(x_size / 2)): x_size + 1 + np.int(np.fix((x_size - 1) / 2)),
    y_size - np.int(np.fix(y_size / 2)): y_size + 1 + np.int(np.fix((y_size - 1) / 2))] = SIM_image_1st * SIM_image_0th

    # Compute crosscorrelation and locate the peak
    CC = high_pass_filter(torch.from_numpy(CC))
    CC = fftshift(fft2(CC))  # Calculate cross - correlation

    CC_peak_loc = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)

    # Obtain shift in original pixel grid from the position of the
    # crosscorrelation peak

    x_size_mid, y_size_mid = x_size_large // 2, y_size_large // 2

    x_size, y_size = CC.shape

    row_shift = CC_peak_loc[0] - x_size_mid
    col_shift = CC_peak_loc[1] - y_size_mid

    row_shift = row_shift / 2
    col_shift = col_shift / 2

    row_shift_sub_pixel, col_shift_sub_pixel = subpixel_register(SIM_image_1st, SIM_image_0th, row_shift, col_shift,
                                                                 usfac=100)

    return np.array([row_shift_sub_pixel, col_shift_sub_pixel])


def subpixel_register(SIM_image_1st, SIM_image_0th, row_shift, col_shift, usfac=100):
    x_size, y_size = SIM_image_1st.shape

    row_shift_upsample = row_shift * usfac
    col_shift_upsample = col_shift * usfac
    # Matrix multiply  DFT around the  current  shift  estimate
    CC = np.conj(dftups(SIM_image_1st * np.conj(SIM_image_0th), usfac, row_shift_upsample, col_shift_upsample))
    # Locate maximum and map back to original  pixel  grid
    rloc, cloc = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)
    rloc = rloc - np.floor((x_size - 1) / 2)
    cloc = cloc - np.floor((y_size - 1) / 2)
    row_shift_result = row_shift - rloc / usfac
    col_shift_result = col_shift - cloc / usfac

    return row_shift_result, col_shift_result


def dftups(CC_in_spatial, usfac=100, roff=0, coff=0):
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
    # Compute kernels and obtain DFT by matrix products
    # kernc = np.exp((-1j * 2 * math.pi / (nc * usfac)) * np.matmul(np.linspace(0,nc-1,nc).reshape(nc,1), ( np.linspace(0,nc-1,noc) - np.floor((nc - 1) / 2) - coff ).reshape(1,noc) ) )
    # kernr = np.exp((-1j * 2 * math.pi / (nr * usfac)) * np.matmul( (np.linspace(0,nor-1,nor).reshape(nor,1) -  roff), np.linspace(0,nr-1,nr).reshape(1,nr) ) )
    # kernc = np.exp((-1j * 2 * math.pi / (nc * usfac)) * np.matmul(ifftshift(np.linspace(0, nc - 1, nc).reshape(nc, 1))- np.floor((nc-1)/2),
    #                                                               (np.linspace(0, noc - 1, noc) - coff).reshape(1,noc)))
    # kernr = np.exp((-1j * 2 * math.pi / (nr * usfac)) * np.matmul((np.linspace(0, nor - 1, nor).reshape(nor, 1) - roff),
    #                                                               ifftshift(np.linspace(0, nr - 1, nr)).reshape(1, nr)- np.floor((nr-1)/2)))

    kernc = np.exp((-1j * 2 * math.pi / (nc * usfac)) * np.matmul(np.linspace(0, nc - 1, nc).reshape(nc, 1), (
            np.linspace(0, nc - 1, nc) - np.floor((nc - 1) / 2) - coff).reshape(1, nc)))
    kernr = np.exp((-1j * 2 * math.pi / (nr * usfac) * np.matmul(
        (np.linspace(0, nr - 1, nr).reshape(nr, 1) - np.floor((nr - 1) / 2) - roff),
        np.linspace(0, nr - 1, nr).reshape(1, nr))))

    CC = np.matmul(np.matmul(kernr, CC_in_spatial), kernc)
    experimental_params = funcs.SinusoidalPattern(probability=1, image_size=nr)
    OTF = experimental_params.OTF_form(fc_ratio=2)
    psf = np.abs(fftshift(ifft2(ifftshift(OTF))))
    OTF_subpixel = np.matmul(np.matmul(kernr, psf), kernc)
    CC /= np.abs(OTF_subpixel)
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

def calculate_m_regression(SIM_data, WF, spatial_frequency, phase):
    """
    :param SIM_data:
    :param WF: wide field image
    :param spatial_frequency:
    :param phase:
    :return: modulation factor m
    """


def calculate_modulation(SIM_data, wide_field_image_direction, estimated_spatial_frequency, estimated_phase):
    """
    Problem: The linear equation made from two raw data with pi interval are not independent!
    calculate the modulation depth. Firstly to separate the frequency component
    :param SIM_data:
    :param wide_field_image_direction:
    :param pixel_frequency: The estimated spatial frequency of sinusoidal pattern
    :return: the modulation depth m
    """
    SIM_data = SIM_data.squeeze()
    SIM_data1 = (SIM_data[0, :, :] - wide_field_image_direction).numpy()
    SIM_data2 = (SIM_data[1, :, :] - wide_field_image_direction).numpy()

    image_size = SIM_data1.shape[0]

    padding_size_one = image_size // 2
    padding_size_two = (image_size + 1) // 2

    phase1 = estimated_phase[0]
    phase2 = estimated_phase[1]

    param_matrix = torch.tensor([[torch.exp(-1j*phase1),torch.exp(1j*phase1)],[torch.exp(-1j*phase2),torch.exp(1j*phase2)]])
    param_matrix = param_matrix.numpy()

    SIM_data1_fft = fftshift(fft2(SIM_data1))
    SIM_data2_fft = fftshift(fft2(SIM_data2))

    param_matrix_inv = np.linalg.inv(param_matrix)

    neg_1_order = param_matrix_inv[0, 0] * SIM_data1_fft + param_matrix_inv[0, 1] * SIM_data2_fft
    pos_1_order = param_matrix_inv[1, 0] * SIM_data1_fft + param_matrix_inv[1, 1] * SIM_data2_fft

    WF_numpy = wide_field_image_direction.numpy()
    WF_fft = fftshift(fft2(WF_numpy))

    experimental_parameters = SinusoidalPattern(probability=1, image_size=image_size)
    if experimental_parameters.upsample == True:
        OTF = experimental_parameters.OTF_upsmaple.numpy()
        CTF = experimental_parameters.CTF_upsmaple.numpy()
        neg_1_order = np.pad(neg_1_order, (padding_size_one, padding_size_two), 'constant')
        pos_1_order = np.pad(pos_1_order, (padding_size_one, padding_size_two), 'constant')
        WF_fft = np.pad(WF_fft, (padding_size_one, padding_size_two), 'constant')
        image_size = image_size * 2

    else:
        OTF = experimental_parameters.OTF.numpy()
        CTF = experimental_parameters.CTF.numpy()

    xx, yy, _, _ = experimental_parameters.GridGenerate(experimental_parameters.upsample, grid_mode='pixel')

    C_psf = fftshift(ifft2(ifftshift(CTF)))

    psf_times_pos_phase_grad = C_psf * np.exp(1j * 2 * math.pi * (
                estimated_spatial_frequency[0] / image_size  * xx + estimated_spatial_frequency[
            1] / image_size  * yy).numpy())
    pos_shift_CTF = abs(fftshift(fft2(psf_times_pos_phase_grad)) )
    pos_shift_CTF[np.where(pos_shift_CTF<0.5)] = 0
    overlap_region1 = pos_shift_CTF * CTF

    psf_times_neg_phase_grad = C_psf * np.exp(-1j * 2 * math.pi * (
                estimated_spatial_frequency[0] / image_size  * xx + estimated_spatial_frequency[
            1] / image_size  * yy).numpy())
    neg_shift_CTF = abs(fftshift(fft2(psf_times_neg_phase_grad)))
    neg_shift_CTF[np.where(neg_shift_CTF < 0.5)] = 0
    overlap_region2 = neg_shift_CTF * CTF

    # weiner_param = 0.04
    # neg_1_order_deconv = neg_1_order * OTF / (OTF**2 + weiner_param)
    # pos_1_order_deconv = pos_1_order * OTF / (OTF**2 + weiner_param)
    #
    # WF_fft_deconv = WF_fft * OTF / (OTF**2 + weiner_param)

    overlap_region1_index = np.where(overlap_region1 > 0.8)
    overlap_region2_index = np.where(overlap_region2 > 0.8)

    '利用衰减的方式：'
    # a = neg_1_order[overlap_region1_index] * OTF[overlap_region2_index]
    # b = WF_fft[overlap_region2_index] * OTF[overlap_region1_index]
    # c = np.mean( abs(a)/abs(b) )

    shift_OTF_pos = shift_fre_sepctrum_V1(OTF, estimated_spatial_frequency, xx, yy)
    shift_OTF_neg = shift_fre_sepctrum_V1(OTF, -estimated_spatial_frequency, xx, yy)

    shift_WF_fft_pos = shift_fre_sepctrum_V1(WF_fft, estimated_spatial_frequency, xx, yy,model = 'image')
    WF_fft_overlap_region1 = abs(shift_WF_fft_pos) * abs(OTF)
    pos_1_order_overlap_region1 = abs(shift_OTF_pos) * abs(pos_1_order)

    shift_WF_fft_neg = shift_fre_sepctrum_V1(WF_fft, -estimated_spatial_frequency, xx, yy,model = 'image')
    WF_fft_overlap_region2 = abs(shift_WF_fft_neg) * abs(OTF)
    neg_1_order_overlap_region2 = abs(shift_OTF_neg) * abs(neg_1_order)

    # m = np.mean(pos_1_order_overlap_region1[overlap_region2_index]) / np.mean(WF_fft_overlap_region1[overlap_region2_index]) + np.mean(neg_1_order_overlap_region2[overlap_region1_index]) / np.mean(WF_fft_overlap_region2[overlap_region1_index])
    m = np.mean(pos_1_order_overlap_region1 * overlap_region2) / np.mean(WF_fft_overlap_region1 * overlap_region2) + np.mean(neg_1_order_overlap_region2 * overlap_region1) / np.mean(WF_fft_overlap_region2 * overlap_region1)
    # m = 0.5 * ( np.mean(abs(neg_1_order_deconv[overlap_region1_index]) / abs(WF_fft_deconv[overlap_region2_index]) ) +
    #             np.mean(abs(pos_1_order_deconv[overlap_region2_index]) / abs(WF_fft_deconv[overlap_region1_index])) )

    print("m = %f" % (m))

    # m = 1


    return m


def calculate_modulation_factor_overlap_region(SIM_first_order, WF_image, estimated_spatial_frequency):
    """
    calculate the modulation depth by comparing the value in the overlapped region
    :param SIM_first_order: The SIM image that only contains +- first orders frequency components
    :param WF_image: The wide field image
    :param estimated_spatial_frequency: The estimated spatial frequency of sinusoidal pattern
    :return: the modulation depth m
    """
    SIM_first_order = SIM_first_order.detach().numpy()
    WF_image = WF_image.detach().numpy()
    image_size = SIM_first_order.shape[0]
    padding_size_one = image_size // 2
    padding_size_two = (image_size + 1) // 2

    SIM_first_order_fft = fftshift(fft2(SIM_first_order, axes=(0, 1)), axes=(0, 1))
    WF_image_fft = fftshift(fft2(WF_image, axes=(0, 1)), axes=(0, 1))

    experimental_parameters = SinusoidalPattern(probability=1, image_size=image_size)
    OTF = experimental_parameters.OTF.numpy()
    CTF = experimental_parameters.CTF.numpy()
    OTF = np.pad(OTF, (padding_size_one, padding_size_two), 'constant')
    psf = fftshift(ifft2(ifftshift(OTF)))

    WF_image_fft_padding = np.pad(WF_image_fft, (padding_size_one, padding_size_two), 'constant')
    SIM_first_order_fft_padding = np.pad(SIM_first_order_fft, (padding_size_one, padding_size_two), 'constant')
    CTF_padding = np.pad(CTF, (padding_size_one, padding_size_two), 'constant')
    experimental_parameters = SinusoidalPattern(probability=1, image_size=image_size * 2)
    xx, yy, _, _ = experimental_parameters.GridGenerate(False, grid_mode='pixel')
    psf_times_phase_gradient_positive = psf * np.exp(-1j * 2 * math.pi * (
                estimated_spatial_frequency[0] / image_size / 2 * xx + estimated_spatial_frequency[
            1] / image_size / 2 * yy).numpy())
    translated_OTF_positive = fftshift(fft2(psf_times_phase_gradient_positive, axes=(0, 1)), axes=(0, 1))
    shift_CTF_positive = np.roll(np.roll(CTF_padding, -1 * int(np.round(estimated_spatial_frequency[0])), 0),
                                 -1 * int(np.round(estimated_spatial_frequency[1])), 1)
    translated_OTF_positive_filtered = translated_OTF_positive * shift_CTF_positive
    psf_times_phase_gradient_negetive = psf * np.exp(1j * 2 * math.pi * (
                estimated_spatial_frequency[0] / image_size / 2 * xx + estimated_spatial_frequency[
            1] / image_size / 2 * yy).numpy())
    translated_OTF_negetive = fftshift(fft2(psf_times_phase_gradient_negetive, axes=(0, 1)), axes=(0, 1))
    shift_CTF_negetive = np.roll(np.roll(CTF_padding, int(np.round(estimated_spatial_frequency[0])), 0),
                                 int(np.round(estimated_spatial_frequency[1])), 1)
    translated_OTF_negetive_filtered = translated_OTF_negetive * shift_CTF_negetive
    overlap_region_WF_sum = np.sum(WF_image_fft_padding * translated_OTF_positive_filtered * CTF_padding)
    overlap_region_first_order_sum = np.sum(
        SIM_first_order_fft_padding * translated_OTF_negetive_filtered * CTF_padding)

    overlap_region_WF_sum1 = np.sum(WF_image_fft_padding * translated_OTF_negetive_filtered * CTF_padding)
    overlap_region_first_order_sum1 = np.sum(
        SIM_first_order_fft_padding * translated_OTF_positive_filtered * CTF_padding)

    m = abs(overlap_region_first_order_sum / overlap_region_WF_sum) + abs(
        overlap_region_first_order_sum1 / overlap_region_WF_sum1)

    return m


def calculate_modulation_factor_using_CC_peak(SIM_image, estimated_spatial_frequency):
    """
    calculate the modulation depth by calculating the CC
    :param SIM_first_order: The SIM image that only contains +- first orders frequency components
    :param WF_image: The wide field image
    :param estimated_spatial_frequency: The estimated spatial frequency of sinusoidal pattern
    :return: the modulation depth m
    “failure
    """
    SIM_image = SIM_image.detach().numpy()
    image_size = SIM_image.shape[0]

    SIM_image_fft = fftshift(fft2(SIM_image, axes=(0, 1)), axes=(0, 1))

    experimental_parameters = SinusoidalPattern(probability=1, image_size=image_size)
    OTF = experimental_parameters.OTF.numpy()
    CTF = experimental_parameters.CTF.numpy()
    # OTF = np.pad(OTF, (padding_size_one, padding_size_two), 'constant')
    psf = fftshift(ifft2(ifftshift(OTF)))

    # WF_image_fft_padding = np.pad(WF_image_fft, (padding_size_one, padding_size_two), 'constant')
    # SIM_first_order_fft_padding = np.pad(SIM_first_order_fft, (padding_size_one, padding_size_two), 'constant')
    # CTF_padding = np.pad(CTF, (padding_size_one, padding_size_two), 'constant')
    experimental_parameters = SinusoidalPattern(probability=1, image_size=image_size)
    xx, yy, _, _ = experimental_parameters.GridGenerate(False, grid_mode='pixel')

    shifted_SIM_fre_spectrum = shift_fre_sepctrum(SIM_image, estimated_spatial_frequency, xx, yy)
    shifted_OTF_postive = shift_fre_sepctrum(psf, estimated_spatial_frequency, xx, yy)
    shifted_OTF_negetive = shift_fre_sepctrum(psf, -1 * estimated_spatial_frequency, xx, yy)
    CC1 = sum(sum(shifted_SIM_fre_spectrum * SIM_image_fft * shifted_OTF_postive))
    CC2 = sum(sum(SIM_image_fft * SIM_image_fft * shifted_OTF_negetive))

    m = 2 * abs(CC1) / abs(CC2)

    return m


def shift_fre_sepctrum(image, shift_vector, xx, yy):
    """
    shift the frequency spectrum of input image in subpixel precision by multiplying gradient matrix in spatial domain
    :param image:
    :param shift_vector: the subpixel shift_vector
    :param xx: the gird matrix in x direction
    :param yy: the gird matrix in y direction
    :return: shifted frequency spectrum
    """
    image_size = image.shape[0]
    image_x_gradient_matrix = image * np.exp(-1j * 2 * math.pi * (
            shift_vector[0] / image_size / 2 * xx + shift_vector[
        1] / image_size / 2 * yy).numpy())
    shifted_image_fre_spectrum = fftshift(fft2(image_x_gradient_matrix, axes=(0, 1)), axes=(0, 1))

    return shifted_image_fre_spectrum

def shift_fre_sepctrum_V1(fre_sepctrum, shift_vector, xx, yy, model = 'OTF'):
    """
    shift the frequency spectrum of input image in subpixel precision by multiplying gradient matrix in spatial domain
    :param image:
    :param shift_vector: the subpixel shift_vector
    :param xx: the gird matrix in x direction
    :param yy: the gird matrix in y direction
    :return: shifted frequency spectrum
    """
    image_size = fre_sepctrum.shape[0]
    if model == 'OTF':
        image = fftshift(ifft2(ifftshift(fre_sepctrum)))
    else:
        image = abs(ifft2(ifftshift(fre_sepctrum)))
    image_times_gradient_matrix = image * np.exp(-1j * 2 * math.pi * (
            shift_vector[0] / image_size * xx + shift_vector[
        1] / image_size* yy).numpy())
    shifted_image_fre_spectrum = fftshift(fft2(image_times_gradient_matrix))

    return shifted_image_fre_spectrum


if __name__ == '__main__':
    pass
    # SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern_and_parameters_of_multichannels(input_SIM_raw_data)
