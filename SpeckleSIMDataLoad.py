#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/16

import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torch.utils.data import DataLoader
from math import pi
# from Add_Sinpattern_OTF_noise_ByAugumentor import SinusoidalPattern

class SIM_data(data.Dataset):
    def __init__(self,
                 directory_data_file,
                 data_mode = 'input_SIM_and_LR_images'
                 ):
        # data_dict=[]
        # with open(directory_data_file,'r',encoding='utf8') as json_file:
        #     for line in json_file.readlines():
        #         dic = json.loads(line)
        #         data_dict.append(dic)
        #     print(data_dict)
        with open(directory_data_file, 'r') as txtFile:
            self.content = txtFile.readlines()

        self.data_mode = data_mode


    def __getitem__(self, index):
        txt_line = self.content[index]
        SIM_data_directory = txt_line.split()[0]
        image_number = int(txt_line.split()[1])
        image_format = txt_line.split()[2]

        LR_image_directoty = SIM_data_directory \
                             + "_LR_" \
                             + '.' + image_format

        HR_image_directoty = SIM_data_directory \
                             + "_SR_" \
                             + '.' + image_format

        HR_image_PIL = Image.open(HR_image_directoty)
        LR_image_PIL = Image.open(LR_image_directoty)

        if len(HR_image_PIL.size) == 2:
            image_size = [HR_image_PIL.size[0],HR_image_PIL.size[1]]
        elif len(HR_image_PIL.size) == 3:
            image_size = [HR_image_PIL.size[1], HR_image_PIL.size[2]]


        transform = transforms.Compose(
            [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        HR_normalized_image_tensor = transform(HR_image_PIL)[0,:,:]
        LR_normalized_image_tensor = transform(LR_image_PIL)[0, :, :]


        SIM_image_data = torch.zeros(image_number+1, image_size[0], image_size[1])

        for i in range(image_number):
            SIM_data_image_directoty = SIM_data_directory \
                                       + "_Speckle_SIM_data(" \
                                       + str(i + 1) \
                                       + ")_" \
                                       + '.' + image_format
            SIM_image_PIL = Image.open(SIM_data_image_directoty)
            SIMdata_normalized_image_tensor = transform(SIM_image_PIL)
            SIM_image_data[i,:,:] = SIMdata_normalized_image_tensor[0,:,:]
            # SIM_image_data[i, :, :] = torch.zeros_like(SIMdata_normalized_image_tensor[0,:,:])


        if self.data_mode == 'input_SIM_and_LR_images':
            SIM_image_data[image_number, :, :] = LR_normalized_image_tensor  # directly use the LR image as input
        elif self.data_mode == 'input_SIM_and_sum_images':
            appro_widefield_image = SIM_image_data.mean(0)
            SIM_image_data[image_number, :,
            :] = appro_widefield_image  # add one layer of approximately wide field image(mean of all the SIM)

        return SIM_image_data,torch.stack((HR_normalized_image_tensor,LR_normalized_image_tensor),2)

    def __len__(self):
        return len(self.content)

if __name__ == '__main__':
    directory_json_file = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\directories_of_images.json"
    directory_txt_file = 'D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test/valid.txt'
    SIM_dataset=SIM_data(directory_txt_file)
    a,b = SIM_dataset[0]
    SIM_data_loader= DataLoader(SIM_dataset, batch_size=4,shuffle=True)