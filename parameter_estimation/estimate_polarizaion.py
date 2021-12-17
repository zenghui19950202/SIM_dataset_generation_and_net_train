import torch
from parameter_estimation import estimate_SIM_pattern
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import common_utils
from self_supervised_learning_sr import forward_model
import torch.optim as optim
import torch.nn as nn


def calculate_polarization_ratio(SIM_raw_data, experimental_parameters,deconv = True):
    if SIM_raw_data.size()[1]==9:
        SIM_data_three_direction = torch.stack(
            [SIM_raw_data[:, 0, :, :], SIM_raw_data[:, 3, :, :], SIM_raw_data[:, 6, :, :]], dim=1)
        _, estimated_pattern_parameters, _ = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
            SIM_data_three_direction, experimental_parameters)
        # print(estimated_pattern_parameters)
        theta = torch.atan2(estimated_pattern_parameters[:, 1], estimated_pattern_parameters[:, 0])
        wide_field_1 = torch.mean(SIM_raw_data[0, 0:3, :, :], dim=0)
        wide_field_2 = torch.mean(SIM_raw_data[0, 3:6, :, :], dim=0)
        wide_field_3 = torch.mean(SIM_raw_data[0, 6:9, :, :], dim=0)
    elif SIM_raw_data.size()[1] == 6:
        SIM_data_three_direction = torch.stack(
            [SIM_raw_data[:, 0, :, :], SIM_raw_data[:, 2, :, :], SIM_raw_data[:, 4, :, :]], dim=1)
        _, estimated_pattern_parameters, _ = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
            SIM_data_three_direction, experimental_parameters)
        # print(estimated_pattern_parameters)
        theta = torch.atan2(estimated_pattern_parameters[:, 1], estimated_pattern_parameters[:, 0])
        wide_field_1 = torch.mean(SIM_raw_data[0, 0:2, :, :], dim=0)
        wide_field_2 = torch.mean(SIM_raw_data[0, 2:4, :, :], dim=0)
        wide_field_3 = torch.mean(SIM_raw_data[0, 4:6, :, :], dim=0)
    wide_field_1, wide_field_2, wide_field_3 = wide_field_1 / wide_field_1.max(), wide_field_2 / wide_field_2.max(), wide_field_3 / wide_field_3.max()
    wide_field_np_1, wide_field_np_2, wide_field_np_3 = wide_field_1.numpy(), wide_field_2.numpy(), wide_field_3.numpy()
    fft_wide_field_np_1, fft_wide_field_np_2, fft_wide_field_np_3 = np.fft.fftshift(
        np.fft.fft2(wide_field_np_1)), np.fft.fftshift(np.fft.fft2(wide_field_np_2)), np.fft.fftshift(
        np.fft.fft2(wide_field_np_3))
    M_matrix = np.array([[1, 1 / 2 * np.exp(1j * 2 * theta[0]), 1 / 2 * np.exp(-2 * 1j * theta[0])],
                         [1, 1 / 2 * np.exp(2 * 1j * theta[1]), 1 / 2 * np.exp(-2 * 1j * theta[1])],
                         [1, 1 / 2 * np.exp(2 * 1j * theta[2]), 1 / 2 * np.exp(-2 * 1j * theta[2])]])
    M_matrix_inv = np.linalg.inv(M_matrix)
    fft_wide_field_np = M_matrix_inv[0, 0] * fft_wide_field_np_1 + M_matrix_inv[0, 1] * fft_wide_field_np_2 + \
                        M_matrix_inv[0, 2] * fft_wide_field_np_3

    wide_field_np = abs(np.fft.ifft2(np.fft.ifftshift(fft_wide_field_np)))
    wide_field = torch.from_numpy(wide_field_np)

    wide_field = torch.mean(SIM_raw_data[:, :, :, :], dim=1)

    OTF = experimental_parameters.OTF
    wide_field_deconv = forward_model.wiener_deconvolution(wide_field, OTF)
    # wide_field_deconv += 1e-3 #add a small number to avoid divide zero
    wide_field_1_deconv = forward_model.wiener_deconvolution(wide_field_1, OTF)
    wide_field_2_deconv = forward_model.wiener_deconvolution(wide_field_2, OTF)
    wide_field_3_deconv = forward_model.wiener_deconvolution(wide_field_3, OTF)
    if wide_field_np.min() < 1e-6:
        wide_field_np += 1e-6
    if SIM_raw_data.size()[1] == 9:

        if deconv:
            polarization_raio = torch.stack(
                [wide_field_1_deconv / (wide_field_deconv), wide_field_1_deconv / (wide_field_deconv),
                 wide_field_1_deconv / (wide_field_deconv), \
                 wide_field_2_deconv / (wide_field_deconv), wide_field_2_deconv / (wide_field_deconv),
                 wide_field_2_deconv / (wide_field_deconv), \
                 wide_field_3_deconv / (wide_field_deconv), wide_field_3_deconv / (wide_field_deconv),
                 wide_field_3_deconv / (wide_field_deconv)],
                dim=0)
            # polarization_raio = torch.stack(
            #     [wide_field_1_deconv / (wide_field_deconv), wide_field_1_deconv / (wide_field_deconv),
            #      wide_field_1_deconv / (wide_field_deconv), \
            #      wide_field_2_deconv / (wide_field_deconv), wide_field_2_deconv / (wide_field_deconv),
            #      wide_field_2_deconv / (wide_field_deconv), \
            #      wide_field_3_deconv / (wide_field_deconv), wide_field_3_deconv / (wide_field_deconv),
            #      wide_field_3_deconv / (wide_field_deconv)],
            #     dim=0)

            # polarization_raio = torch.stack([wide_field_1_deconv / (wide_field_deconv**2+1e-6), wide_field_1_deconv / (wide_field_deconv**2+1e-6), wide_field_1_deconv / (wide_field_deconv**2+1e-6), \
            #                                  wide_field_2_deconv / (wide_field_deconv**2+1e-6), wide_field_2_deconv / (wide_field_deconv**2+1e-6), wide_field_2_deconv / (wide_field_deconv**2+1e-6), \
            #                                  wide_field_3_deconv / (wide_field_deconv**2+1e-6), wide_field_3_deconv / (wide_field_deconv**2+1e-6), wide_field_3_deconv / (wide_field_deconv**2+1e-6)],
            #                                 dim=0)
        else:
            polarization_raio = torch.stack([wide_field_1 / wide_field, wide_field_1 / wide_field, wide_field_1 / wide_field, \
                                             wide_field_2 / wide_field, wide_field_2 / wide_field, wide_field_2 / wide_field, \
                                             wide_field_3 / wide_field, wide_field_3 / wide_field, wide_field_3 / wide_field],
                                            dim=0)

        index = polarization_raio > polarization_raio.mean() * 3
        polarization_raio[index]=1
    elif SIM_raw_data.size()[1] == 6:
        polarization_raio = torch.stack([wide_field_1 / wide_field, wide_field_1 / wide_field, \
                                         wide_field_2 / wide_field, wide_field_2 / wide_field,  \
                                         wide_field_3 / wide_field, wide_field_3 / wide_field],
                                        dim=0)
        # polarization_raio = torch.stack([wide_field_1_deconv / wide_field_deconv, wide_field_1_deconv / wide_field_deconv, \
        #                                  wide_field_2_deconv / wide_field_deconv, wide_field_2_deconv / wide_field_deconv, \
        #                                  wide_field_3_deconv / wide_field_deconv, wide_field_3_deconv / wide_field_deconv],dim=0)
    return polarization_raio.unsqueeze(0)


def calculate_polarization_direction(SIM_raw_data, experimental_parameters):
    if SIM_raw_data.size()[1]==9:
        SIM_data_three_direction = torch.stack(
            [SIM_raw_data[:, 0, :, :], SIM_raw_data[:, 3, :, :], SIM_raw_data[:, 6, :, :]], dim=1)
        _, estimated_pattern_parameters, _ = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
            SIM_data_three_direction, experimental_parameters)
        # print(estimated_pattern_parameters)
        theta = torch.atan2(estimated_pattern_parameters[:, 0], estimated_pattern_parameters[:, 1])
        wide_field_1 = torch.mean(SIM_raw_data[0, 0:3, :, :], dim=0)
        wide_field_2 = torch.mean(SIM_raw_data[0, 3:6, :, :], dim=0)
        wide_field_3 = torch.mean(SIM_raw_data[0, 6:9, :, :], dim=0)
    else:
        SIM_data_three_direction = torch.stack(
            [SIM_raw_data[:, 0, :, :], SIM_raw_data[:, 2, :, :], SIM_raw_data[:, 4, :, :]], dim=1)
        _, estimated_pattern_parameters, _ = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
            SIM_data_three_direction, experimental_parameters)
        # print(estimated_pattern_parameters)
        theta = torch.atan2(estimated_pattern_parameters[:, 0], estimated_pattern_parameters[:, 1])
        wide_field_1 = torch.mean(SIM_raw_data[0, 0:2, :, :], dim=0)
        wide_field_2 = torch.mean(SIM_raw_data[0, 2:4, :, :], dim=0)
        wide_field_3 = torch.mean(SIM_raw_data[0, 4:6, :, :], dim=0)

    wide_field_1_deconv = forward_model.wiener_deconvolution(wide_field_1, experimental_parameters.OTF)
    wide_field_2_deconv = forward_model.wiener_deconvolution(wide_field_2, experimental_parameters.OTF)
    wide_field_3_deconv = forward_model.wiener_deconvolution(wide_field_3, experimental_parameters.OTF)
    wide_field_1_deconv, wide_field_2_deconv, wide_field_3_deconv = wide_field_1_deconv / wide_field_1_deconv.max(), wide_field_2_deconv / wide_field_2_deconv.max(), wide_field_3_deconv / wide_field_3_deconv.max()
    wide_field_np_1, wide_field_np_2, wide_field_np_3 = wide_field_1_deconv.numpy(), wide_field_2_deconv.numpy(), wide_field_3_deconv.numpy()
    wide_field_1, wide_field_2, wide_field_3 = wide_field_1 / wide_field_1.max(), wide_field_2 / wide_field_2.max(), wide_field_3 / wide_field_3.max()
    wide_field_np_1, wide_field_np_2, wide_field_np_3 = wide_field_1.numpy(), wide_field_2.numpy(), wide_field_3.numpy()
    fft_wide_field_np_1, fft_wide_field_np_2, fft_wide_field_np_3 = np.fft.fftshift(
        np.fft.fft2(wide_field_np_1)), np.fft.fftshift(np.fft.fft2(wide_field_np_2)), np.fft.fftshift(
        np.fft.fft2(wide_field_np_3))
    M_matrix = np.array([[1, -1 / 2 * np.exp(1j * 2 * theta[0]), -1 / 2 * np.exp(-2 * 1j * theta[0])],
                         [1, -1 / 2 * np.exp(2 * 1j * theta[1]), -1 / 2 * np.exp(-2 * 1j * theta[1])],
                         [1, -1 / 2 * np.exp(2 * 1j * theta[2]), -1 / 2 * np.exp(-2 * 1j * theta[2])]])
    # M_matrix = np.array([[1, 1 / 2 * np.exp(1j * 2 * theta[0]), 1 / 2 * np.exp(-2 * 1j * theta[0])],
    #                      [1, 1 / 2 * np.exp(2 * 1j * theta[1]), 1 / 2 * np.exp(-2 * 1j * theta[1])],
    #                      [1, 1 / 2 * np.exp(2 * 1j * theta[2]), 1 / 2 * np.exp(-2 * 1j * theta[2])]])
    M_matrix_inv = np.linalg.inv(M_matrix)

    polarization_fft = M_matrix_inv[2, 0] * fft_wide_field_np_1 + M_matrix_inv[2, 1] * fft_wide_field_np_2 + \
                       M_matrix_inv[2, 2] * fft_wide_field_np_3

    OTF_np = experimental_parameters.OTF.numpy()
    polarization = np.fft.ifft2(np.fft.ifftshift(polarization_fft*OTF_np / (OTF_np*OTF_np+0.04)))
    polarization_angle_np = np.angle(polarization) / 2
    polarization_angle = torch.from_numpy(polarization_angle_np)
    absorption_efficiency = 1 + 0.4 * torch.cos(2 * ( theta.reshape([1, 1, 3])-polarization_angle.unsqueeze(2)))
    polarization_raio = torch.stack(
        [absorption_efficiency[:, :, 0], absorption_efficiency[:, :, 0], absorption_efficiency[:, :, 0],
         absorption_efficiency[:, :, 1], absorption_efficiency[:, :, 1], absorption_efficiency[:, :, 1], \
         absorption_efficiency[:, :, 2], absorption_efficiency[:, :, 2], absorption_efficiency[:, :, 2]],
        dim=0)
    polarization_angle= -polarization_angle
    alpha = polarization_angle % math.pi
    h = alpha / math.pi
    # s = 0.6 * torch.ones_like(h)
    # v = wide_field / wide_field.max()
    # RGB = HSV2BGR(torch.stack([h,s,v],2))
    # common_utils.save_image_tensor2pillow(RGB.unsqueeze(0),'/data/zh/self_supervised_learning_SR/test_for_self_9_frames_supervised_SR_net/color_wheel/')
    # plt.imshow(RGB)
    # plt.show()
    # dipole_direction = calculate_dipole_direction(wide_field_123, theta)
    return polarization_raio.unsqueeze(0), h


def calculate_polarization_ratio_regression(SIM_raw_data, experimental_parameters):
    SIM_data_three_direction = torch.stack(
        [SIM_raw_data[:, 0, :, :], SIM_raw_data[:, 3, :, :], SIM_raw_data[:, 6, :, :]], dim=1)
    _, estimated_pattern_parameters, _ = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
        SIM_data_three_direction, experimental_parameters)
    # print(estimated_pattern_parameters)
    wide_field_direction = torch.zeros_like(SIM_data_three_direction).squeeze()
    theta = torch.atan2(estimated_pattern_parameters[:, 0], estimated_pattern_parameters[:, 1])
    wide_field_direction[0, :, :] = torch.mean(SIM_raw_data[0, 0:3, :, :], dim=0)
    wide_field_direction[1, :, :] = torch.mean(SIM_raw_data[0, 3:6, :, :], dim=0)
    wide_field_direction[2, :, :] = torch.mean(SIM_raw_data[0, 6:9, :, :], dim=0)
    wide_field_direction[0, :, :], wide_field_direction[1, :, :], wide_field_direction[2, :, :] = wide_field_direction[
                                                                                                  0, :,
                                                                                                  :] / wide_field_direction[
                                                                                                       0, :,
                                                                                                       :].max(), wide_field_direction[
                                                                                                                 1, :,
                                                                                                                 :] / wide_field_direction[
                                                                                                                      1,
                                                                                                                      :,
                                                                                                                      :].max(), wide_field_direction[
                                                                                                                                2,
                                                                                                                                :,
                                                                                                                                :] / wide_field_direction[
                                                                                                                                     2,
                                                                                                                                     :,
                                                                                                                                     :].max()
    wide_field_np_1, wide_field_np_2, wide_field_np_3 = wide_field_direction[0, :, :].numpy(), wide_field_direction[1,
                                                                                               :,
                                                                                               :].numpy(), wide_field_direction[
                                                                                                           2, :,
                                                                                                           :].numpy()
    fft_wide_field_np_1, fft_wide_field_np_2, fft_wide_field_np_3 = np.fft.fftshift(
        np.fft.fft2(wide_field_np_1)), np.fft.fftshift(np.fft.fft2(wide_field_np_2)), np.fft.fftshift(
        np.fft.fft2(wide_field_np_3))
    M_matrix = np.array([[1, 1 / 2 * np.exp(1j * 2 * theta[0]), 1 / 2 * np.exp(-2 * 1j * theta[0])],
                         [1, 1 / 2 * np.exp(2 * 1j * theta[1]), 1 / 2 * np.exp(-2 * 1j * theta[1])],
                         [1, 1 / 2 * np.exp(2 * 1j * theta[2]), 1 / 2 * np.exp(-2 * 1j * theta[2])]])
    M_matrix_inv = np.linalg.inv(M_matrix)
    fft_wide_field_np = M_matrix_inv[0, 0] * fft_wide_field_np_1 + M_matrix_inv[0, 1] * fft_wide_field_np_2 + \
                        M_matrix_inv[0, 2] * fft_wide_field_np_3

    wide_field_np = abs(np.fft.ifft2(np.fft.ifftshift(fft_wide_field_np)))
    wide_field = torch.from_numpy(wide_field_np).float()
    if wide_field_np.min() < 1e-6:
        wide_field_np += 1e-6

    params = []
    polarization_raio = torch.zeros_like(SIM_raw_data[0, 0:3, :, :])
    polarization_raio.requires_grad = True
    params += [{'params': polarization_raio}]

    optimizer_dipole_direction = optim.Adam(params, lr=0.005)
    MSE_loss = nn.MSELoss()
    num_epochs = 600
    device = 'cpu'
    OTF = experimental_parameters.OTF
    # calculate the deconvolution of averaged wide_field image
    wide_field_complex = torch.stack(
        [wide_field.squeeze(), torch.zeros_like(wide_field).squeeze()], 2)
    wide_field_fft_deconv = forward_model.torch_2d_fftshift(
        torch.fft((wide_field_complex), 2)) / (OTF.unsqueeze(2) + 0.04)
    wide_field_deconv = torch.ifft(forward_model.torch_2d_ifftshift(wide_field_fft_deconv), 2)

    # calculate the Fourier transform of wide_field image at each direction
    image_size = wide_field_direction.size()
    wide_field_direction_fft = torch.zeros([image_size[0], image_size[1], image_size[2], 2])
    for i in range(3):
        wide_field_direction_complex = torch.stack(
            [wide_field_direction[i, :, :].squeeze(), torch.zeros_like(wide_field).squeeze()], 2)
        wide_field_direction_fft[i, :, :, :] = forward_model.torch_2d_fftshift(
            torch.fft((wide_field_direction_complex), 2))

    # regress the polarization_raio by minimizing the difference between estimated fft of wide_field image and measured fft of wide field image
    for epoch in range(num_epochs):
        loss = torch.tensor([0.0], dtype=torch.float32, device=device)
        optimizer_dipole_direction.zero_grad()
        for i in range(polarization_raio.size()[0]):
            estimated_wide_field_direction = wide_field_deconv * polarization_raio[i, :, :].clip(0, 2).unsqueeze(2)
            estimated_wide_field_direction_fft = forward_model.torch_2d_fftshift(
                torch.fft((estimated_wide_field_direction), 2)) * OTF.unsqueeze(2)
            mse_loss = MSE_loss(estimated_wide_field_direction_fft, wide_field_direction_fft[i, :, :, :])
            loss += mse_loss
        loss.backward()
        optimizer_dipole_direction.step()

        with torch.no_grad():
            train_loss = loss.float()

        print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))
    polarization_raio = abs(polarization_raio.detach())
    # index = polarization_raio > polarization_raio.mean() * 3
    # polarization_raio[index] = 1
    result = torch.stack(
        [polarization_raio[0, :, :], polarization_raio[0, :, :], polarization_raio[0, :, :],
         polarization_raio[1, :, :], polarization_raio[1, :, :], polarization_raio[1, :, :],
         polarization_raio[2, :, :], polarization_raio[2, :, :], polarization_raio[2, :, :]], 0)

    return result.detach().unsqueeze(0)


def HSV2BGR(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h = h * 6.0
    k = torch.floor(h)

    p = h - k
    t = 1 - s
    n = 1 - s * p
    p = 1 - (s * (1 - p))

    kc = ((k == 0) | (k == 6))
    r = kc
    g = kc * p
    b = kc * t

    kc = (k == 1)
    r = r + kc * n
    g = g + kc
    b = b + kc * t

    kc = (k == 2)
    r = r + kc * t
    g = g + kc
    b = b + kc * p

    kc = (k == 3)
    r = r + kc * t
    g = g + kc * n
    b = b + kc

    kc = (k == 4)
    r = r + kc * p
    g = g + kc * t
    b = b + kc

    kc = (k == 5)
    r = r + kc
    g = g + kc * t
    b = b + kc * n

    RGB = torch.stack([r, g, b], 2)
    RGB = (v / RGB.max()).unsqueeze(2) * RGB
    return RGB

# def calculate_dipole_direction_iterative(polarization_raio, theta):
#     polarization_raio = polarization_raio.squeeze()
#     polarization_raio = polarization_raio.to(device)
#     params = []
#     dipole_direction = torch.zeros_like(polarization_raio[0, :, :])
#     theta = theta.to(device)
#     dipole_direction.requires_grad = True
#
#     polarization_modulator = torch.tensor([0.7]).to(device)
#     polarization_modulator.requires_grad = True
#     polarization_phi = torch.tensor([0.0]).to(device)
#     polarization_phi.requires_grad = True
#
#     params += [{'params': dipole_direction, 'weight_decay': weight_decay}]
#     params += [{'params': polarization_modulator, 'weight_decay': weight_decay}]
#     params += [{'params': polarization_phi, 'weight_decay': weight_decay}]
#     optimizer_dipole_direction = optim.Adam(params, lr=0.005)
#     MSE_loss = nn.MSELoss()
#
#     for epoch in range(2000):
#         loss = torch.tensor([0.0], dtype=torch.float32, device=device)
#         optimizer_dipole_direction.zero_grad()
#         for i in range(polarization_raio.size()[0]):
#             absorption_efficiency = 1 - polarization_modulator * (
#                 torch.cos(2 * (theta[math.floor(i / 3)] - dipole_direction) + polarization_phi))
#             absorption_efficiency = absorption_efficiency / absorption_efficiency.max()
#             mse_loss = MSE_loss(absorption_efficiency, polarization_raio[i, :, :] / polarization_raio[i, :, :].max())
#             loss += mse_loss
#         loss.backward()
#         optimizer_dipole_direction.step()
#
#         with torch.no_grad():
#             train_loss = loss.float()
#
#         print('epoch: %d/%d, train_loss: %f' % (epoch + 1, num_epochs, train_loss))
#     plt.imshow(dipole_direction.detach().cpu())
#     plt.show()
#     return dipole_direction
#
#
# def calculate_dipole_direction(wide_field_123, theta):
#     # 使用三步移项法来解调相位
#     dipole_direction = torch.zeros_like(wide_field_123[0, :, :])
#     real_part = wide_field_123[0, :, :] * torch.cos(theta[0]) + wide_field_123[1, :, :] * torch.cos(
#         theta[1]) + wide_field_123[2, :, :] * torch.cos(theta[2])
#     imag_part = wide_field_123[0, :, :] * torch.sin(theta[0]) + wide_field_123[1, :, :] * torch.sin(
#         theta[1]) + wide_field_123[2, :, :] * torch.sin(theta[2])
#     dipole_direction = torch.atan2(imag_part,real_part)
#     plt.imshow(dipole_direction.detach().cpu())
#     plt.show()
#     return dipole_direction
