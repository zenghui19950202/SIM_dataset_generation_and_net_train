#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/8

import torch
import math
import Pipeline_speckle
from torchvision import transforms
from fuctions_for_generate_pattern import SinusoidalPattern
from Augmentor.Operations import Crop
from configparser import ConfigParser
import random


if __name__ == '__main__':

    config = ConfigParser()
    config.read('configuration.ini')
    config.sections()
    SourceFileDirectory = config.get('image_file', 'SourceFileDirectory')
    sample_num = config.getint('SIM_data_generation', 'sample_num')
    image_size = config.getint('SIM_data_generation', 'image_size')
    data_ratio = config.getfloat('SIM_data_generation', 'data_ratio')
    # SourceFileDirectory = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/test2"

    # p = Pipeline_speckle.Pipeline_speckle(source_directory=SourceFileDirectory)
    # p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    # p.add_operation(SpecklePattern(probability=1,image_size=image_size))
    # p.sample(20,multi_threaded=True,data_ratio=1)

    train_directory = SourceFileDirectory + '/train'
    valid_directory = SourceFileDirectory + '/valid'

    p = Pipeline_speckle.Pipeline_speckle(source_directory=train_directory,output_directory="train")
    p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    p.add_operation(SinusoidalPattern(probability=1,image_size=image_size))
    p.sample(10,multi_threaded=True,data_type='train',data_num=9)

    p = Pipeline_speckle.Pipeline_speckle(source_directory=valid_directory,output_directory="valid")
    p.add_operation(Crop(probability=1, width = image_size, height = image_size, centre = False))
    p.add_operation(SinusoidalPattern(probability=1,image_size=image_size))
    p.sample(10,multi_threaded=True,data_type='valid',data_num=9)