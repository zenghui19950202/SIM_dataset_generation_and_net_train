#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/28

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 函数功能：用于搜索并插值计算结构光的最高空间频率
# 函数参数：image待计算图像，fx频率横坐标网格，fy频率纵坐标网格
# 函数输出：函数的空间频率（fx，fy）
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from self_supervised_learning_sr import common_utils

def CalculateSpatialFrequency1 (image, fx, fy,resolution,PixleSize):
    image_np = common_utils.torch_to_np(image)


    imageF = fftshift(fft2(image))
    imageF = abs(imageF)
    ImageSize = max(max(size(image)))
    CutoffRadius = 0.3 * floor(PixleSize / resolution * ImageSize)
    temp = FindPeak(image, CutoffRadius)
    initPos(2) = temp(1)
    initPos(1) = temp(2)

    step = 1e-2 # 沿方向搜索计算时的步距，这里的单位是像素而不是空间频率
    tolerance = 1e-7 # 计算结果的最高精度，比给出结果的有效位数高一个小数位
    grad = zeros(1, 2) # 存储计算的梯度（dx，dy）
    P0 = GetP(imageF, fx, fy, GetSpatialFrequency(fx, fy, initPos)) # 获取峰值的函数
    n = 0 # 循环的计数器
    while (step > tolerance) and (n < 10):
        grad = GetGrad(imageF, fx, fy, initPos, tolerance) # 计算梯度
        P1 = GetP(imageF, fx, fy, GetSpatialFrequency(fx, fy, initPos + grad * step)) # 重新计算峰值，要在负梯度位置上找
        if P1 >= P0:
            P0 = P1 # 更新峰值强度
            initPos = initPos + grad * step # 更新位置
        else:
            step = step * 0.1
        n = n + 1
    # % 1/norm(GetSpatialFrequency (fx, fy, initPos))
    output = GetSpatialFrequency(fx, fy, initPos);
    # 在这里输出的是应该是无量纲的量，即不带实际尺寸单位的频率