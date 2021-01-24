#!/usr/bin/env python
# -*- coding: utf-8 -*-
# draw intensity profiles using csv data exported from ImageJ
# author：zenghui time:2020/11/27

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalization(line_intensity):
    line_intensity_sub_bg = line_intensity - line_intensity.min()
    line_intensity_normalized = line_intensity_sub_bg/ line_intensity_sub_bg.max()

    return line_intensity_normalized


csv_directory = '/home/common/Zenghui/test_draw_plot/plot_csv/2'
data_3_frame = pd.read_csv(csv_directory + '/3.csv')
distance = data_3_frame['X'].values
line_intensity_3 = data_3_frame['Y'].values

line_intensity_5 = pd.read_csv(csv_directory + '/5.csv')['Y'].values
line_intensity_9 = pd.read_csv(csv_directory + '/9.csv')['Y'].values
line_intensity_LR = pd.read_csv(csv_directory + '/LR.csv')['Y'].values
line_intensity_HR = pd.read_csv(csv_directory + '/HR.csv')['Y'].values

plt.plot(distance,normalization(line_intensity_3),label='3 frame',linewidth=3,linestyle='-.')
plt.plot(distance,normalization(line_intensity_5),label='5 frame',linewidth=3)
plt.plot(distance,normalization(line_intensity_9),label='9 frame',linewidth=3)
plt.plot(distance,normalization(line_intensity_LR),label='LR',linewidth=3,linestyle='--')
plt.plot(distance,normalization(line_intensity_HR),label='HR',linewidth=3)

ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 0))
# ax.spines['right'].set_position(('data', distance[-2]))
# plt.xticks(distance[1:])
plt.legend()
plt.savefig(csv_directory + 'resolution_plot.eps', dpi=600,format='eps')
# plt.yticks(distance[1:])
plt.show()








