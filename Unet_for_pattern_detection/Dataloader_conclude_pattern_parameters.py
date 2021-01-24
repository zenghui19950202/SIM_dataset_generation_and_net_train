#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/12/25

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/3
from parameter_estimation import *
import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torch.utils.data import DataLoader
from math import pi
from utils import *
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern
# from Add_Sinpattern_OTF_noise_ByAugumentor import SinusoidalPattern

class SIMdata_pattern_pairs(data.Dataset):
    def __init__(self,
                 directory_data_file,
                 ):
        # data_dict=[]
        # with open(directory_data_file,'r',encoding='utf8') as json_file:
        #     for line in json_file.readlines():
        #         dic = json.loads(line)
        #         data_dict.append(dic)
        #     print(data_dict)
        with open(directory_data_file, 'r') as txtFile:
            self.content = txtFile.readlines()


    def __getitem__(self, index):
        txt_line = self.content[index]
        # for i in txt_line:
        #     i = i.strip('\n')
        #     i = str(i)
        #     if i.count('-') == 1:
        #         print(i)
        SIM_data_directory = txt_line.split()[0]
        image_number = int(txt_line.split()[1])
        image_format = txt_line.split()[2]
        SpatialFrequencyX = float(txt_line.split()[3])
        SpatialFrequencyY = float(txt_line.split()[4])
        phase = float(txt_line.split()[5])
        modulation_factor = float(txt_line.split()[6])

        pattern_parameters = [SpatialFrequencyX,SpatialFrequencyY,phase,modulation_factor]

        SIMdata_directoty = SIM_data_directory \
                             + "_SIMdata_" \
                             + '.' + image_format

        pattern_directoty = SIM_data_directory \
                             + "_pattern_" \
                             + '.' + image_format

        SIMdata_PIL = Image.open(SIMdata_directoty)
        pattern_PIL = Image.open(pattern_directoty)

        if len(SIMdata_PIL.size) == 2:
            image_size = [SIMdata_PIL.size[0],SIMdata_PIL.size[1]]
        elif len(SIMdata_PIL.size) == 3:
            image_size = [SIMdata_PIL.size[1], SIMdata_PIL.size[2]]


        transform = transforms.Compose(
            [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        SIMdata_image_tensor = transform(SIMdata_PIL)[0,:,:]
        pattern_image_tensor = transform(pattern_PIL)[0, :, :]

        return SIMdata_image_tensor.unsqueeze(0),pattern_parameters

    def __len__(self):
        return len(self.content)

if __name__ == '__main__':
    data_parameters = load_configuration_parameters.load_train_net_config_paras()

    SourceFileDirectory = data_parameters['train_directory_file']

    SIM_dataset=SIMdata_pattern_pairs(SourceFileDirectory)
    num = 0
    error_rate = 0
    experimental_parameters = SinusoidalPattern(probability=1, image_size=512)
    for SIM_data,pattern_parameters in SIM_dataset:
        SIM_data = SIM_data* 0.5 + 0.5
        temp_input_SIM_pattern, estimated_pattern_parameters, estimated_SIM_pattern_without_m = estimate_SIM_pattern.estimate_SIM_pattern_and_parameters_of_multichannels_V1(
            SIM_data.unsqueeze(0))

        pattern_parameters = torch.tensor(pattern_parameters)

        pattern_parameters[0:2] = pattern_parameters[0:2] / experimental_parameters.delta_fx
        temp = pattern_parameters[2].clone()
        pattern_parameters[2] = pattern_parameters[3]
        pattern_parameters[3] = torch.cos(temp)
        estimated_pattern_parameters[0,3] = torch.cos(estimated_pattern_parameters[0,3])

        gt_modulation_factor = pattern_parameters[2]
        estimated_modulation_factor = estimated_pattern_parameters[0, 2]
        loss = abs(gt_modulation_factor - estimated_modulation_factor)
        error_rate += loss / gt_modulation_factor

        num += 1
        even_error_rate = error_rate / num
        print('epoch: %d, error_rate: %f . GT: %f, estimated: %f' % (num, even_error_rate,gt_modulation_factor,estimated_modulation_factor))
        print(pattern_parameters)
        print(estimated_pattern_parameters[0,0:-1])





    # SIM_data_loader= DataLoader(SIM_dataset, batch_size=4,shuffle=True)