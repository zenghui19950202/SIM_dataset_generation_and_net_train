#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/16
import cv2
import torch
import os
import uuid
import numpy as np

def rename_data_for_loading(image_directory,crop_size = 512, data_name = '0000000',LR_name ='UnapodizedReconstruction', HR_name = 'WideField'):
    SIM_data_image_directoty = image_directory + '/' + data_name + '1'+ '.' + 'tif'
    SIM_image_np_1 = cv2.imread(SIM_data_image_directoty, -1) / 1.0
    image_size = [SIM_image_np_1.shape[0], SIM_image_np_1.shape[1]]
    image_number = 9

    save_file = os.path.dirname(image_directory)+'/SIMdata_SR_train'

    if not os.path.exists(save_file):
        os.mkdir(save_file)

    txt_directory = os.path.dirname(image_directory) + '/SIMdata_SR_train.txt'
    f = open(txt_directory, 'a')
    UUID = uuid.uuid4()
    save_directory = save_file + '/'+ str(UUID)+ '_Speckle_SIM_data'
    pattern_save_directory = save_file + '/'+ str(UUID)+ '_Speckle_SIM_pattern'

    f.write(save_file + '/'+ str(UUID) + '\t' + str(image_number) + '\t' + 'tif' + '\n')
    f.close()

    HR_image_directoty = image_directory \
                               +'/'\
                               + LR_name \
                               + '.' + 'tif'
    HR_image_np = cv2.imread(HR_image_directoty, -1)
    save_name = save_file + '/'+ str(UUID) \
                + "_SR_" \
                + '.' + 'tif'
    if HR_image_np.shape[0] > crop_size:
        HR_image_np_crop = crop_center(HR_image_np,crop_size)
        cv2.imwrite(save_name, HR_image_np_crop)
    else:
        cv2.imwrite(save_name, HR_image_np)

    LR_image_directoty = image_directory \
                         + '/' \
                         + HR_name \
                         + '.' + 'tif'
    LR_image_np = cv2.imread(LR_image_directoty, -1)
    save_name = save_file + '/'+ str(UUID) \
                + "_LR_" \
                + '.' + 'tif'
    if LR_image_np.shape[0] > crop_size:
        LR_image_np_crop = crop_center(LR_image_np, crop_size)
        cv2.imwrite(save_name, LR_image_np_crop)
    else:
        cv2.imwrite(save_name, LR_image_np)

    if image_size[0] < crop_size:
        pad_size = crop_size - image_size[0]
        pad_size_one = pad_size//2
        pad_size_two = (pad_size+1)//2
        for i in range(image_number):
            SIM_data_image_directoty = image_directory \
                                       + '/' \
                                       + data_name\
                                       + str(i + 1) \
                                       + '.' + 'tif'
            SIM_image_np = cv2.imread(SIM_data_image_directoty, -1)

            SIM_data_renamed_directoty = save_directory \
                                               + '('\
                                               + str(i + 1) \
                                               + ")_" \
                                               + '.' + 'tif'
            SIM_pattern_renamed_directoty = pattern_save_directory \
                                               + '('\
                                               + str(i + 1) \
                                               + ")_" \
                                               + '.' + 'tif'
            SIM_image_np_pad = np.pad(SIM_image_np,(pad_size_one,pad_size_two),'constant')
            cv2.imwrite(SIM_data_renamed_directoty,SIM_image_np_pad)
            cv2.imwrite(SIM_pattern_renamed_directoty, SIM_image_np_pad)
    else:
        for i in range(image_number):
            SIM_data_image_directoty = image_directory \
                                       + '/' \
                                       + data_name\
                                       + str(i + 1) \
                                       + '.' + 'tif'
            SIM_image_np = cv2.imread(SIM_data_image_directoty, -1)

            SIM_data_renamed_directoty = save_directory \
                                         + '(' \
                                         + str(i + 1) \
                                         + ")_" \
                                         + '.' + 'tif'
            SIM_pattern_renamed_directoty = pattern_save_directory \
                                            + '(' \
                                            + str(i + 1) \
                                            + ")_" \
                                            + '.' + 'tif'
            SIM_image_np_crop = crop_center(SIM_image_np, crop_size)
            cv2.imwrite(SIM_data_renamed_directoty, SIM_image_np_crop)
            cv2.imwrite(SIM_pattern_renamed_directoty, SIM_image_np_crop)
        # SIM_image_data[i, :, :] = torch.zeros_like(SIMdata_normalized_image_tensor[0,:,:])

def crop_center(img,crop_size):
    y,x = img.shape
    startx = x//2 - crop_size//2
    starty = y//2 - crop_size//2
    return img[starty:(starty+crop_size), startx:(startx+crop_size)]
if __name__ == '__main__':
    # rename_data_for_loading('/home/common/zenghui/2/', LR_name ='AVG_raw-530nm', HR_name = 'AVG_raw-530nm' )
    rename_data_for_loading('/data1/zh1/PRSIM/Lal_4SIM/microtublin/',crop_size= 256, data_name ='0000000', LR_name ='WideField', HR_name = 'UnapodizedReconstruction' )