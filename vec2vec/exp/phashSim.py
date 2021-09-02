#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import heapq
import operator
import itertools
import collections
import numpy
import scipy as np
from sklearn.metrics.pairwise import cosine_similarity

# from fancy import *
# from itertools import chain

# def flatten(x):
#     result = []
#     for el in x:
#         if isinstance(x, collections.Iterable) and not isinstance(el, str):
#             result.extend(flatten(el))
#         else:
#             result.append(el)
#     return result

def flatten(input_array):
    result_array = []
    for element in input_array:
        if isinstance(element, float):
            result_array.append(element)
        elif isinstance(element, list):
            result_array += flatten(element)
    return result_array


def pHash(imgfile):
    """get image pHash value"""
    #加载并调整图片为32x32灰度图片
    # img=cv2.imread(imgfile, 0)
    img=imgfile
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)

        #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img       #填充数据

    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    # vis1=np.resize(vis1,(32,32))
    vis1.resize(32, 32)

    #把二维list变成一维list
    img_list=flatten(vis1.tolist())

    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]

    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])



def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])