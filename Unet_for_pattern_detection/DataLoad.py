#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/3

import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torch.utils.data import DataLoader
from math import pi
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
        SIM_data_directory = txt_line.split()[0]
        image_number = int(txt_line.split()[1])
        image_format = txt_line.split()[2]

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

        return SIMdata_image_tensor.unsqueeze(0),pattern_image_tensor.unsqueeze(0)

    def __len__(self):
        return len(self.content)

if __name__ == '__main__':
    directory_json_file = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\directories_of_images.json"
    directory_txt_file = '/data/zh/test/train.txt'


    SIM_dataset=SIMdata_pattern_pairs(directory_txt_file)
    a,b = SIM_dataset[0]
    a = a * 0.5 + 0.5

    a_PIL = transforms.ToPILImage()(a).convert('RGB')
    b = b * 0.5 + 0.5
    b_PIL = transforms.ToPILImage()(b).convert('RGB')


    # SIM_data_loader= DataLoader(SIM_dataset, batch_size=4,shuffle=True)