import torch
from torchvision import transforms
from PIL import Image
from math import pi
import numpy as np
from utils import common_utils
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
from numpy.fft import fft2
from numpy.fft import fftshift
from numpy.fft import ifft2
from numpy.fft import ifftshift
import math


def read_image(path):

    image_PIL = Image.open(path)
    image_PIL = image_PIL.convert('RGB')
    transform = transforms.Compose(
                        [transforms.ToTensor()])
    image_tensor = transform(image_PIL)[0, :, :]

    return image_tensor

def freq_domain_SIM_flower_filter(image,estimated_pattern_params,CTF):
    """
    :param image: SIM SR result image using iterative approach
    :param spatial_freq: the spatial frequency of illumination pattern [3,2]tensor 3 orientations, 2 the x,y coordinate
    :param CTF: the passband of frequency information
    :return: the filtered SIM SR image
    """
    image = image.detach().cpu()
    CTF_np = CTF.detach().cpu().numpy()
    estimated_pattern_params = estimated_pattern_params.detach().cpu()
    image_size = CTF.size()
    shift_CTF = np.zeros([image_size[0],image_size[1],6])
    for i in range(3):
        spatial_freq = estimated_pattern_params[i*2,0:2]
        shift_CTF[:,:,i*2] = np.roll(np.roll(CTF_np, int(np.round(spatial_freq[0])), 0),
                            int(np.round(spatial_freq[1])), 1)
        shift_CTF[:,:,i*2+1] = np.roll(np.roll(CTF_np, -1*int(np.round(spatial_freq[0])), 0),
                            -1*int(np.round(spatial_freq[1])), 1)
    shift_CTF = torch.from_numpy(shift_CTF)
    CTF_sum = torch.sum(shift_CTF, 2)
    CTF_sum += CTF_np
    flower_filter = torch.where(CTF_sum > 0, torch.Tensor([1]), torch.Tensor([0]))
    flower_filter_np = flower_filter.numpy()

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
    dim = CTF.shape

    if len(FFT_NumpyImage.shape) == 3:
        FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) * flower_filter_np.reshape(dim[0], dim[1],
                                                                                                  1)  # reshape the OTF to fullfill the broadcast mechanism
    elif len(FFT_NumpyImage.shape) == 2:
        FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) *flower_filter_np
    else:
        raise Exception('The dimensions of input images must be 2 or 3 ')

    Filter_NumpyImage = np.fft.ifft2(np.fft.ifftshift(FilterFFT_NumpyImage, axes=(0, 1)), axes=(0, 1))
    Filter_NumpyImage = abs(Filter_NumpyImage)

    Filter_NumpyImage = Filter_NumpyImage / Filter_NumpyImage.max() * 256

    filter_tensor_Image = torch.from_numpy(Filter_NumpyImage)

    return filter_tensor_Image

def generlized_wiener_filter(image,estimated_pattern_params):
    image_size = image.size()[0]
    experimental_parameters = SinusoidalPattern(probability=1,image_size = image_size)
    if experimental_parameters.upsample ==True:
        OTF = experimental_parameters.OTF_upsmaple.numpy()
        estimated_spatial_frequency = estimated_pattern_params[:, 0:2].detach().cpu() * 2
    else:
        OTF = experimental_parameters.OTF.numpy()
        estimated_spatial_frequency = estimated_pattern_params[:, 0:2].detach().cpu()


    xx, yy, fx, fy = experimental_parameters.GridGenerate(grid_mode='pixel',
                                                        up_sample=experimental_parameters.upsample)
    fr_square = (xx ** 2 + yy ** 2)
    f0 = image_size / 256 * 2
    notch_filter = 1 - torch.exp(-1 * fr_square / (4 * f0 * f0))

    psf = fftshift(ifft2(ifftshift(OTF * notch_filter.numpy())))
    shifted_OTF = np.zeros_like(psf)
    input_image_num = estimated_spatial_frequency.size()[0]
    for i in range(3):
        num = i * 2
        psf_times_phase_gradient = psf * np.exp(1j * 2 * math.pi * (
                    estimated_spatial_frequency[num,0] / image_size / 2 * xx + estimated_spatial_frequency[
                num,1] / image_size / 2 * yy).numpy())
        shifted_OTF += fftshift(fft2(psf_times_phase_gradient, axes=(0, 1)), axes=(0, 1))

        psf_times_phase_gradient = psf * np.exp(-1j * 2 * math.pi * (
                    estimated_spatial_frequency[num,0] / image_size / 2 * xx + estimated_spatial_frequency[
                num,1] / image_size / 2 * yy).numpy())
        shifted_OTF += fftshift(fft2(psf_times_phase_gradient, axes=(0, 1)), axes=(0, 1))

    shifted_OTF += OTF

    NumpyImage = image.squeeze().numpy()
    FFT_NumpyImage = np.fft.fftshift(np.fft.fft2(NumpyImage, axes=(0, 1)), axes=(0, 1))

    alpha = 1 # winnier parameter
    Filter_FFT_NumpyImage = FFT_NumpyImage / (shifted_OTF + alpha)

    Filter_NumpyImage = np.fft.ifft2(np.fft.ifftshift(Filter_FFT_NumpyImage, axes=(0, 1)), axes=(0, 1))
    Filter_NumpyImage = abs(Filter_NumpyImage)

    Filter_NumpyImage = Filter_NumpyImage / Filter_NumpyImage.max() * 256

    filter_tensor_Image = torch.from_numpy(Filter_NumpyImage)

    return filter_tensor_Image





