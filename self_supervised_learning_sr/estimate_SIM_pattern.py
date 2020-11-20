#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/29
import torch
import math
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from self_supervised_learning_sr import estimate_SIM_pattern_parameters
import torch.nn as nn


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
    estimated_SIM_pattern_parameters = torch.zeros(input_channel, 4)
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size, grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)
    for i in range(input_channel):
        one_channel_SIM_data = SIM_data[:, i, :, :].squeeze()
        estimated_spatial_frequency, estimated_modulation_factor = estimate_SIM_pattern_parameters.calculate_spatial_frequency(
            one_channel_SIM_data * one_channel_SIM_data)
        # estimated_modulation_factor = 1
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data,
                                                                          estimated_spatial_frequency)
        estimated_SIM_pattern_parameters[i, :] = torch.tensor(
            [*estimated_spatial_frequency, estimated_modulation_factor, torch.tensor(estimated_phase)])
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
    return estimated_SIM_pattern, estimated_SIM_pattern_parameters


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


if __name__ == '__main__':
    pass
    # SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern_and_parameters_of_multichannels(input_SIM_raw_data)
