#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/12/29

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/3
import Augmentor
from PIL import Image
import random
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy
from torchvision import transforms
import numpy as np
import math
from simulation_data_generation import Pipeline_speckle
import torch
from parameter_estimation import *
from utils import *
from Augmentor.Operations import Crop
from Unet_for_pattern_detection import generate_sinusoidal_SIMdata_and_pattern_pairs
from Unet_for_pattern_detection import Pipeline_SIMdata_pattern_pairs
import matplotlib.pyplot as plt

class Pipeline_SIMdata_pattern_pairs(Pipeline_speckle.Pipeline_speckle):

    def __init__(self, source_directory=None, output_directory="output",image_size = 256, save_format=None):
       super(Pipeline_SIMdata_pattern_pairs,self).__init__(source_directory=source_directory, output_directory=output_directory,  save_format=save_format)
       self.source_directory=source_directory

       self.train_txt_directory = os.path.dirname(output_directory) + '/SIMdata_SR_train.txt'
       self.valid_txt_directory = os.path.dirname(output_directory) + '/SIMdata_SR_valid.txt'
       self.SNR_level_num = 40
       self.image_size = image_size

       data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
       SourceFileDirectory = data_generation_parameters['SourceFileDirectory']
       self.Magnification = data_generation_parameters['Magnification']
       self.PixelSizeOfCCD = data_generation_parameters['PixelSizeOfCCD']
       self.EmWaveLength = data_generation_parameters['EmWaveLength']
       self.NA = data_generation_parameters['NA']
       self.NumPhase = data_generation_parameters['NumPhase']
       self.SNR = data_generation_parameters['SNR']
       self.photon_num = data_generation_parameters['photon_num']
       self.f_cutoff = 1 / 0.61 * self.NA / self.EmWaveLength  # The coherent cutoff frequency
       self.f_cutoff = 2 * self.NA / self.EmWaveLength  # The coherent cutoff frequency

       if image_size == None:
           self.image_size = data_generation_parameters['image_size']
       else:
           self.image_size = image_size
       self.pattern_frequency_ratio = data_generation_parameters['pattern_frequency_ratio']
       self.data_num = data_generation_parameters['data_num']
       self.PixelSize = self.PixelSizeOfCCD / self.Magnification
       self.delta_x = self.PixelSize  # xy方向的空域像素间隔，单位m
       self.delta_y = self.PixelSize
       self.delta_fx = 1 / self.image_size / self.delta_x  # xy方向的频域像素间隔，单位m ^ -1
       self.delta_fy = 1 / self.image_size / self.delta_y

       if self.PixelSize > 0.61 * self.EmWaveLength / self.NA / 4:
           self.SR_image_size = self.image_size * 2
           self.SR_PixelSize = self.PixelSizeOfCCD / self.Magnification / 2
           self.upsample = True
           self.xx_upsmaple, self.yy_upsmaple, self.fx_upsmaple, self.fy_upsmaple = self.GridGenerate(self.image_size,
                                                                                                      up_sample=self.upsample)
           self.f_upsample = pow((self.fx_upsmaple ** 2 + self.fy_upsmaple ** 2), 1 / 2)
           self.OTF_upsmaple = self.OTF_form(fc_ratio=1, pixel_mode='upsample')
       else:
           self.upsample = False

       self.upsample = False

       self.xx, self.yy, self.fx, self.fy = self.GridGenerate(self.image_size)

       self.f = pow((self.fx ** 2 + self.fy ** 2), 1 / 2)  # The spatial freqneucy fr=sqrt( fx^2 + fy^2 )

       self.OTF = self.OTF_form(fc_ratio=1)
       self.CTF = self.CTF_form(fc_ratio=1)

    def add_poisson_noise(self, tensor_Image,photon_num):  # The type of input image is PIL
        # if len(TensorImage)==3:
        #      TensorImage = TensorImage.permute(1, 2, 0) # transope for matplot
        # signal_intensity_of_image = (tensor_Image ** 2).mean()  # The mean intensity of signal
        np_image = tensor_Image.squeeze().numpy()
        noised_np_image = np.random.poisson(np_image/np_image.max() * photon_num)

        noised_np_image_normalized = noised_np_image / noised_np_image.max()
        tensor_image_with_noise_normalized = torch.from_numpy(noised_np_image_normalized)
        return tensor_image_with_noise_normalized
        # return image_with_noise_normalized

    def GridGenerate(self, image_size=256, up_sample=False, grid_mode='real'):
        '''
        :param Magnification: the magnification of the Microscope
        :param PixelSize: the PixleSize of the sCMOS or CCD
        :param EmWaveLength:  emission wavelength of sample
        :param NA:  NA(numerical aperture) of the objective
        :return:
        '''
        if up_sample == True:
            y, x = self.SR_image_size, self.SR_image_size
        else:
            y, x = self.image_size, self.image_size

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
        if up_sample == False:
            if grid_mode == 'real':
                fx = xx * self.delta_fx
                fy = yy * self.delta_fy
                xx = xx * self.delta_x
                yy = yy * self.delta_y
            elif grid_mode == 'pixel':
                fx = xx * 1.0 / self.image_size
                fy = yy * 1.0 / self.image_size
            else:
                raise Exception('error grid mode')
        else:
            if grid_mode == 'real':
                fx = xx * self.delta_fx
                fy = yy * self.delta_fy
                xx = xx * self.SR_PixelSize
                yy = yy * self.SR_PixelSize
            elif grid_mode == 'pixel':
                fx = xx * 1.0 / self.SR_image_size
                fy = yy * 1.0 / self.SR_image_size
            else:
                raise Exception('error grid mode')

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
            FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) * OTF.numpy().reshape(dim[0],
                                                                                                      dim[1],
                                                                                                      1)  # reshape the OTF to fullfill the broadcast mechanism
        elif len(FFT_NumpyImage.shape) == 2:
            FilterFFT_NumpyImage = np.fft.fftshift(FFT_NumpyImage, axes=(0, 1)) * OTF.numpy()
        else:
            raise Exception('The dimensions of input images must be 2 or 3 ')

        Filter_NumpyImage = np.fft.ifft2(np.fft.ifftshift(FilterFFT_NumpyImage, axes=(0, 1)), axes=(0, 1))
        Filter_NumpyImage = abs(Filter_NumpyImage)

        Filter_NumpyImage = Filter_NumpyImage / Filter_NumpyImage.max() * 256

        filter_tensor_Image = torch.from_numpy(Filter_NumpyImage)
        return filter_tensor_Image

    def OTF_form(self, fc_ratio=1, pixel_mode='default'):
        f0 = fc_ratio * self.f_cutoff
        if pixel_mode == 'default':
            f = self.f
        elif pixel_mode == 'upsample':
            f = self.f_upsample

        OTF = torch.where(f < f0, (2 / math.pi) * (torch.acos(f / f0) - (f / f0) * (
            pow((1 - (f / f0) ** 2), 0.5))), torch.Tensor([0]))  # Caculate the OTF support
        # OTF = torch.where(f < f0,torch.ones_like(f),torch.zeros_like(f))
        return OTF

    def CTF_form(self, fc_ratio=1):
        f0 = fc_ratio * self.f_cutoff / 2
        f = self.f
        CTF = torch.where(f < f0, torch.Tensor([1]), torch.Tensor([0]))
        return CTF

    def psf_form(self, OTF):

        OTF = OTF.squeeze()
        Numpy_OTF = OTF.numpy()
        psf = np.fft.ifftshift(np.fft.ifft2(Numpy_OTF, axes=(0, 1)), axes=(0, 1))
        psf = abs(psf)
        psf_Numpy = psf / psf.max()
        psf_tensor = torch.from_numpy(psf_Numpy)
        half_size_of_psf = int(psf.shape[0] / 2)
        half_row_of_psf = psf_tensor[half_size_of_psf]
        # a = half_row_of_psf < 1e-2
        id = torch.arange(0, half_row_of_psf.nelement())[half_row_of_psf.gt(1e-3)]
        psf_crop = psf_tensor[id[0]:id[-1] + 1, id[0]:id[-1] + 1]
        return psf_crop

    def sample(self, n, multi_threaded=True, data_type='train', data_num=16):
        """
        Generate :attr:`n` number of samples from the current pipeline.

        This function samples from the pipeline, using the original images
        defined during instantiation. All images generated by the pipeline
        are by default stored in an ``output`` directory, relative to the
        path defined during the pipeline's instantiation.

        By default, Augmentor will use multi-threading to increase the speed
        of processing the images. However, this may slow down some
        operations if the images are very small. Set :attr:`multi_threaded`
        to ``False`` if slowdown is experienced.

        :param n: The number of new samples to produce.
        :type n: Integer
        :param multi_threaded: Whether to use multi-threading to process the
         images. Defaults to ``True``.
        :type multi_threaded: Boolean
        :return: None
        """
        augmentor_images = list(range(n))
        self.data_num = data_num
        self.data_type = data_type


        if self.data_type == 'train':
            self.txt_directory = self.train_txt_directory
        elif self.data_type == 'valid':
            self.txt_directory = self.valid_txt_directory
        else:
            raise Exception("error data_type")

        if len(self.augmentor_images) == 0:
            raise IndexError("There are no images in the pipeline. "
                             "Add a directory using add_directory(), "
                             "pointing it to a directory containing images.")

        total_estimated_modulation_factor = []
        total_GT_modulation_factor = []
        total_error_rate = torch.zeros(self.SNR_level_num)
        if n == 0:
            augmentor_images = self.augmentor_images
        else:
            for i in range(n):
                augmentor_images[i] = copy.deepcopy(random.choice(self.augmentor_images))

        if multi_threaded:
            # TODO: Restore the functionality (appearance of progress bar) from the pre-multi-thread code above.
            with tqdm(total=len(augmentor_images), desc="Executing Pipeline", unit=" Samples") as progress_bar:
                with ThreadPoolExecutor(max_workers=None) as executor:
                    for result in executor.map(self, augmentor_images):
                        progress_bar.set_description("Processing %s" % result)
                        progress_bar.update(1)
        else:
            with tqdm(total=len(augmentor_images), desc="Executing Pipeline", unit=" Samples") as progress_bar:
                for augmentor_image in augmentor_images:
                    estimated_modulation_factor,GT_modulation_factor = self._execute(augmentor_image)
                    loss = abs(GT_modulation_factor - estimated_modulation_factor)
                    error_rate = loss / GT_modulation_factor
                    total_error_rate += error_rate
                    total_estimated_modulation_factor.append(estimated_modulation_factor)
                    total_GT_modulation_factor.append(GT_modulation_factor)
                progress_bar.set_description("Processing %s" % os.path.basename(augmentor_image.image_path))
                progress_bar.update(1)

        even_error_rate = total_error_rate / sample_num_train

        print(total_estimated_modulation_factor)
        print(total_GT_modulation_factor)
        torch.save(total_estimated_modulation_factor,SaveFileDirectory + '/even_estimated_modulation_factor')
        torch.save(total_GT_modulation_factor, SaveFileDirectory + '/even_GT_modulation_factor')
        plt.plot(torch.linspace(0,self.SNR_level_num,self.SNR_level_num),even_error_rate.squeeze())
        plt.savefig(SaveFileDirectory + '/error_of_modulation_factor.eps', dpi=600, format='eps')
        plt.show()




    def get_entropy(self,img_):
        tmp = []
        for i in range(256):
            tmp.append(0)
        k = 0
        res = 0
        img_ = img_ /img_.max()*255
        img = np.array(img_,dtype='uint8')
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k = float(k + 1)
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
            if (tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        return res

    def _execute(self, augmentor_image,save_to_disk=True, multi_threaded=True,):
        """
        Private method. Used to pass an image through the current pipeline,
        and return the SIM images, and  write image directories, wave vectors, phi into a json file  .

        The returned image can then either be saved to disk or simply passed
        back to the user. Currently this is fixed to True, as Augmentor
        has only been implemented to save to disk at present.

        :param augmentor_image: The image to pass through the pipeline.
        :param save_to_disk: Whether to save the image to disk. Currently
         fixed to true.
        :type augmentor_image: :class:`ImageUtilities.AugmentorImage`
        :type save_to_disk: Boolean
        :return: The augmented image.
        """

        images = []

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        # What if they are array data?
        if augmentor_image.pil_images is not None:
            images.append(augmentor_image.pil_images)

        if augmentor_image.ground_truth is not None:
            if isinstance(augmentor_image.ground_truth, list):
                for image in augmentor_image.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(augmentor_image.ground_truth))


        try:
            estimated_modulation_factor_array = torch.zeros(self.SNR_level_num)
            GT_modulation_factor = torch.zeros(self.SNR_level_num)
            for i in range(self.SNR_level_num):
                image_pair,parameters = self.perform_operation(images[0],i)
                SIM_data_PIL = image_pair[0]
                SIM_data = transforms.ToTensor()(SIM_data_PIL.convert('L'))
                pattern_parameters = parameters
                _, estimated_pattern_parameters, _ = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
                    SIM_data.unsqueeze(0))

                pattern_parameters = torch.tensor(pattern_parameters)
                gt_modulation_factor = pattern_parameters[3]
                estimated_modulation_factor = estimated_pattern_parameters[0, 2]

                estimated_modulation_factor_array[i] = estimated_modulation_factor.clone()
                GT_modulation_factor[i] = gt_modulation_factor.clone()
                print('SNR_level_num: %d , estimated_modulation_factor: %f, gt_modulation_factor : %f' %(i,estimated_modulation_factor,gt_modulation_factor) )

        except IOError as e:
            print("Error writing %s, %s. Change save_format to PNG?" % ('LR', e.message))
            print("You can change the save format using the set_save_format(save_format) function.")
            print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")


        return estimated_modulation_factor_array,GT_modulation_factor

    def perform_operation(self, image,SNR_level_num):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.

        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        crop_size=self.image_size
        h, w = image.size
        pad_w = max(crop_size - w, 0)
        pad_h = max(crop_size - h, 0)
        img_pad = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(image)
        center_crop = transforms.CenterCrop(size=(crop_size, crop_size))
        imag_pad_crop = center_crop(img_pad)

        # augmented_images += self.LR_image_generator(imag_pad_crop)
        image_gray = imag_pad_crop.convert('L')
        TensorImage = transforms.ToTensor()(image_gray)

        image,parameters = self.SinusoidalPattern(TensorImage,SNR_level_num)

        return image,parameters

    def SinusoidalPattern(self, image,SNR_level_num):
        '''
        :param image:  PIL_Image that will be loaded pattern on
        :param NumPhase:  Number of phase
        :return: SinusoidalPatternImage: Image which loaded sinusoidal pattern
        '''
        resolution = 0.61 * self.EmWaveLength / self.NA
        # xx, yy, _, _ = self.GridGenerate(image=torch.rand(7, 7))
        # xx, yy, fx, fy = self.GridGenerate(image)
        # TensorImage = transforms.ToTensor()(image)

        random_theta = random.random() * 2 * math.pi
        pattern_frequency_ratio = random.random() / 2 + 0.5

        SpatialFrequencyX = -pattern_frequency_ratio * 1 / resolution * math.sin(random_theta)  # 0.8倍的极限频率条纹 pattern_frequency_ratio，可调
        SpatialFrequencyY = -pattern_frequency_ratio * 1 / resolution * math.cos(random_theta)

        random_phase = random.random() * 2 * math.pi
        random_modulation = 0.05 + random.random() * 0.95
        SinPattern = (torch.cos(
            random_phase + 2 * math.pi * (SpatialFrequencyX * self.xx + SpatialFrequencyY * self.yy)) * random_modulation + 1) / 2
        SinPattern_OTF_filter = self.OTF_Filter(SinPattern * image,self.OTF)
        photon_num = pow(10, 5 / self.SNR_level_num * SNR_level_num + 1)
        SinPattern_OTF_filter_poisson_noise = self.add_poisson_noise(SinPattern_OTF_filter,photon_num)

        SinPattern_OTF_filter_poisson_noise_PIL = transforms.ToPILImage()(SinPattern_OTF_filter_poisson_noise).convert('RGB')
        SinPattern_PIL_ = transforms.ToPILImage()(SinPattern).convert('RGB')

        return [SinPattern_OTF_filter_poisson_noise_PIL,SinPattern_PIL_],[SpatialFrequencyX,SpatialFrequencyY,random_phase,random_modulation]




if __name__ == '__main__':
    data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()

    SourceFileDirectory = data_generation_parameters['SourceFileDirectory']
    SaveFileDirectory = data_generation_parameters['save_file_directory']

    EmWaveLength = data_generation_parameters['EmWaveLength']
    NA = data_generation_parameters['NA']

    train_directory = SourceFileDirectory + '/train'
    valid_directory = SourceFileDirectory + '/valid'


    image_size = data_generation_parameters['image_size']
    data_num = data_generation_parameters['data_num']
    sample_num_train = data_generation_parameters['sample_num_train']
    sample_num_valid = data_generation_parameters['sample_num_valid']


    # TODO:output_directory换成自己定义的目录
    p = Pipeline_SIMdata_pattern_pairs(source_directory=train_directory, output_directory=SaveFileDirectory+'/SIMdata_SR_train',image_size = image_size)
    p.sample(sample_num_train,multi_threaded=False,data_type='train', data_num=1)

