#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/9

import os

if __name__ == '__main__':

    libs = {"torchsummary" , "Augmentor ","opencv-python"}
    try:
        for lib in libs:
            os.system(" pip install " + lib)
            print("{}   Install successful".format(lib))
    except:
        print("{}   failed install".format(lib))