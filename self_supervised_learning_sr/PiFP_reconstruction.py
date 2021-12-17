"""
pattern-illuminated Fourier Ptychography reconstruction
"""
from self_supervised_learning_sr import forward_model
from utils import common_utils
import copy
from torch.utils.data import DataLoader
from simulation_data_generation import fuctions_for_generate_pattern as funcs
from utils import *
import torch
from parameter_estimation import *
import numpy as np
import matplotlib.pyplot as plt
import imageio

def PiFP_recon(SIM_data, SIM_pattern, sample, experimental_params,save_gif = False):
    epoch = 200  # temp number
    SIM_data_num = SIM_data.size()[0]
    if experimental_params.upsample == True:
        OTF = experimental_params.OTF_upsmaple.numpy()
    else:
        OTF = experimental_params.OTF.numpy()

    SIM_data = SIM_data.numpy()
    SIM_pattern = SIM_pattern.numpy()
    sample = sample.numpy()

    save_file_directory = '/data/zh/self_supervised_learning_SR/test_for_self_9_frames_supervised_SR_net/resolution_verify_board/normal/fre_ratio_0.8_6frame/'

    image_list = []
    for i in range(epoch):
        for j in range(SIM_data_num):
            SIM_data_est = OTF_Filter(sample * SIM_pattern[j, :, :], OTF)
            delta = OTF_Filter(SIM_data[j,:,:] - SIM_data_est, OTF )
            temp_sample = sample + delta * SIM_pattern[j, :, :] / (SIM_pattern[j, :, :].max() ** 2)
            # SIM_pattern[j, :, :] += sample / (sample.max() ** 2) * delta
            sample = temp_sample
        # plt.imshow(sample)
        # plt.savefig(save_file_directory+'temp.png')
        # image_list.append(imageio.imread(save_file_directory+'temp.png'))
        print("PiFP iterative epoch: %d" % i )

    # imageio.mimsave(save_file_directory+'pic.gif', image_list, duration=0.5)

    plt.imshow(sample, cmap='gray')
    plt.show()

    return sample


def OTF_Filter(image, OTF):

    FFT_NumpyImage = np.fft.fft2(image)
    FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage) * OTF

    Filter_NumpyImage = np.fft.ifft2(np.fft.ifftshift(FilterFFT_NumpyImage))
    Filter_NumpyImage = Filter_NumpyImage.real

    return Filter_NumpyImage


if __name__ == '__main__':
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    train_directory_file = train_net_parameters['train_directory_file']
    valid_directory_file = train_net_parameters['valid_directory_file']
    save_file_directory = train_net_parameters['save_file_directory']
    data_generate_mode = train_net_parameters['data_generate_mode']
    net_type = train_net_parameters['net_type']
    data_input_mode = train_net_parameters['data_input_mode']
    LR_highway_type = train_net_parameters['LR_highway_type']
    MAX_EVALS = train_net_parameters['MAX_EVALS']
    num_epochs = train_net_parameters['num_epochs']
    data_num = train_net_parameters['data_num']
    image_size = train_net_parameters['image_size']
    opt_over = train_net_parameters['opt_over']

    SIM_data = SpeckleSIMDataLoad.SIM_data_load(train_directory_file, normalize=True, data_mode='only_raw_SIM_data')
    SIM_pattern = SpeckleSIMDataLoad.SIM_pattern_load(train_directory_file, normalize=True)
    # SIM_pattern = SIM_data_load(train_directory_file, normalize=False, data_mode='only_raw_SIM_data')

    SIM_data_dataloader = DataLoader(SIM_data, batch_size=1)
    SIM_pattern_dataloader = DataLoader(SIM_pattern, batch_size=1)

    input_id = 1
    data_id = 0
    for SIM_data_one_group, _ in zip(SIM_data_dataloader, SIM_pattern_dataloader):
        # SIM_raw_data = SIM_data[0]
        if data_id == input_id:
            break
        data_id += 1

    SIM_raw_data = SIM_data_one_group[0]

    image_size = [SIM_raw_data.size()[2], SIM_raw_data.size()[3]]
    experimental_params = funcs.SinusoidalPattern(probability=1, image_size=image_size[0]) #

    LR_HR = SIM_data_one_group[1]
    HR = LR_HR[:, :, :, 0]

    wide_field_image = torch.mean(SIM_raw_data[:, :, :, :], dim=1)
    wide_field_image = wide_field_image / wide_field_image.max()
    LR = torch.mean(SIM_raw_data[:, :, :, :], dim=1).unsqueeze(0)

    if experimental_params.upsample == True:
        up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        SR_image = up_sample(copy.deepcopy(wide_field_image).unsqueeze(0))
        input_SIM_raw_data = up_sample(SIM_raw_data)
        HR = up_sample(HR.unsqueeze(0))
        LR = up_sample(LR)
        # SR_image = torch.zeros_like(SR_image)
    else:
        SR_image = copy.deepcopy(wide_field_image).unsqueeze(0)
        input_SIM_raw_data = SIM_raw_data
    HR = HR / HR.max()
    HR = HR.squeeze().numpy() * 255

    "extract the illumination parameters"
    temp_input_SIM_pattern, estimated_pattern_parameters, estimated_SIM_pattern_without_m = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
        SIM_raw_data, experimental_params)

    # temp_input_SIM_pattern, estimated_pattern_parameters = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_TIRF_image(SIM_raw_data, SIM_raw_data, experimental_params)
    print(estimated_pattern_parameters)
    SIM_raw_data = SIM_raw_data.cpu().squeeze()
    SIM_pattern = temp_input_SIM_pattern.cpu().squeeze()
    input_SIM_raw_data = input_SIM_raw_data.cpu().squeeze()
    LR = LR.cpu().squeeze()
    PiFP_recon(input_SIM_raw_data, SIM_pattern, LR, experimental_params)