#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/29
import torch
import math
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from self_supervised_learning_sr import estimate_SIM_pattern_parameters


def estimate_SIM_pattern_and_parameters_of_multichannels(SIM_data):
    batch_size, input_channel, _, _ = SIM_data.shape
    estimated_SIM_pattern = torch.zeros_like(SIM_data)
    experimental_parameters = SinusoidalPattern(probability=1)
    image_size = experimental_parameters.image_size
    estimated_SIM_pattern_parameters = torch.zeros(input_channel,4)
    xx, yy, _, _ = experimental_parameters.GridGenerate(image_size, grid_mode='pixel')
    for i in range(input_channel):
        one_channel_SIM_data = SIM_data[:, i, :, :].squeeze()
        estimated_spatial_frequency, estimated_modulation_factor = estimate_SIM_pattern_parameters.calculate_spatial_frequency(
            one_channel_SIM_data * one_channel_SIM_data)
        estimated_phase = estimate_SIM_pattern_parameters.calculate_phase(one_channel_SIM_data,
                                                                          estimated_spatial_frequency)
        estimated_SIM_pattern_parameters[i,:] = torch.tensor([*estimated_spatial_frequency,estimated_modulation_factor,torch.tensor(estimated_phase)])
        estimated_SIM_pattern[:, i, :, :] = (estimated_modulation_factor * torch.cos(
            estimated_phase + 2 * math.pi * (
                        estimated_spatial_frequency[0] * xx / image_size + estimated_spatial_frequency[
                    1] * yy / image_size)) + 1) / 2
    return estimated_SIM_pattern,estimated_SIM_pattern_parameters

def fine_adjust_SIM_pattern(SIM_data_shape,intial_estimated_pattern_params,delta_pattern_params,xx,yy):

    pattern_params = intial_estimated_pattern_params + delta_pattern_params
    estimated_SIM_pattern = torch.zeros(SIM_data_shape)
    channels = SIM_data_shape[1]
    image_size = SIM_data_shape[2]

    for i in range(channels):
        estimated_spatial_frequency_x = pattern_params[i][0]
        estimated_spatial_frequency_y = pattern_params[i][1]
        estimated_modulation_factor = pattern_params[i][2]
        estimated_phase = pattern_params[i][3]

        estimated_SIM_pattern[:, i, :, :] = (estimated_modulation_factor * torch.cos(
            estimated_phase + 2 * math.pi * (
                    estimated_spatial_frequency_x * xx / image_size + estimated_spatial_frequency_y * yy / image_size)) + 1) / 2

    return  estimated_SIM_pattern
