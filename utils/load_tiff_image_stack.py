#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/31
import cv2

if __name__ == '__main__':
    image_stack_tif = "/home/common/zenghui/2/raw-530nm.tif"
    SIM_raw_data_stack = cv2.imread(image_stack_tif, -1) / 1.0
    a = SIM_raw_data_stack