#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/9

import os

libs = {"torchsummary" , "Augmentor "}
try:
    for lib in libs:
        os.system(" pip install " + lib)
        print("{}   Install successful".format(lib))
except:
    print("{}   failed install".format(lib))