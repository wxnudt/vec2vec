#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

from plotnine import *
import pandas as pd
from plotnine.themes.elements import *

# #分别存放所有点的横坐标和纵坐标，一一对应
# x1_list = [128, 256, 400, 512, 640]
# x2_list = [128, 256, 400, 512, 600]
# m2v_list = [82.3, 95.38, 80.4, 80.2, 80.4]
# lle_list = [72.0, 74.1, 73.7, 72.65, 70.8]
# le_list = [69.65, 69.65, 69.05, 68.35, 65.7]
# pca_list = [79.75, 81.55, 82.55, 82.65, 83.1]
# d2v_list = [85.65, 87.4, 87.15, 87.00, 86.7]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.plot(x1_list, lle_list, color='green', label='LLE', marker='.')
# plt.plot(x1_list, le_list, color='cyan', label='LE', marker='o')
# plt.plot(x1_list, pca_list, color='skyblue', label='PCA', marker='v')
# plt.plot(x1_list, d2v_list, color='blue', label='doc2vec', marker='+')
# plt.plot(x1_list, m2v_list, color='red', label='Our Method', marker='x')
# plt.legend(fontsize=14)  # 显示图例
# plt.ylim((40,100))
# plt.xlabel('Embedding Dimensions',fontsize=14)
# plt.ylabel('Accuracy',fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.tight_layout()
# plt.grid()
# plt.show()


# # Text classification running time
# x1_list = [100, 1000, 5000, 10000, 12000]
# m2v_list = [4.49405, 29.730265, 144.782905, 304.066387, 376.457795]
# lle_list = [1.011182, 17.048905, 315.084032, 1287.955079, 1878.316312]
# le_list = [0.320568, 13.457025, 283.360880, 1081.625459, 1471.574146]
# pca_list = [0.795412, 2.650757, 8.661204, 15.750142, 22.746649]
# d2v_list = [3.797227, 4.322377, 15.608411, 24.032324, 28.659635]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.plot(x1_list, lle_list, color='green', label='LLE', marker='.')
# plt.plot(x1_list, le_list, color='cyan', label='LE', marker='o')
# plt.plot(x1_list, pca_list, color='skyblue', label='PCA', marker='v')
# plt.plot(x1_list, d2v_list, color='blue', label='doc2vec', marker='+')
# plt.plot(x1_list, m2v_list, color='red', label='Our Method', marker='x')
#
# plt.legend(fontsize=14)
# plt.xlabel('Number of data',fontsize=14)
# plt.ylabel('Running Time (s)',fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.grid()
# plt.show()



# # Image classification running time on SVHN dataset. the dimension of y is 3072
# x1_list = [100, 1000, 5000, 10000, 15000, 20000, 25000, 30000]
# m2v_list = [1.575796, 11.061574, 55.707760, 113.673277, 175.364460, 243.697148, 315.971394, 396.403634]
# lle_list = [0.710489, 2.640389, 34.918028, 147.377760, 488.856008, 822.258403, 1304.385407, 3884.911357]
# le_list = [0.212076, 1.572422, 30.886781, 121.850163, 277.678371, 665.431856, 935.358672, 2101.290721]
# umap_list = [8.80548, 9.958134, 22.063326, 24.943131, 28.136431, 38.594092, 50.720222, 63.836295]
# Avec_list = [1.858840, 10.610689, 51.735917, 106.554751, 161.463905, 216.925684, 275.622544, 337.825940]
#
# #isomap_list = [0.221583, 14.805410, 66.054196, 148.831012, 681.764511, 1611.054285, 2198.848976, 4706.658612]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.plot(x1_list, lle_list, color='green', label='LLE', marker='v')
# plt.plot(x1_list, le_list, color='cyan', label='LE', marker='o')
# #plt.plot(x1_list, isomap_list, color='blue', label='Isomap', marker='+')
# plt.plot(x1_list, umap_list, color='gray', label='UMAP', marker='>')
# plt.plot(x1_list, m2v_list, color='red', label='Vec2vec', marker='x')
# plt.plot(x1_list, Avec_list, color='blue', label='AVec2vec', marker='d')
#
# plt.legend(fontsize=14)
# plt.xlabel('Number of data',fontsize=14)
# plt.ylabel('Running Time (s)',fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.tight_layout()
# plt.grid()
# plt.show()



# # Image classification running time on SVHN dataset. the number of data point is 2000
# x1_list = [100, 1000, 5000, 10000, 50000, 100000, 150000, 200000, 250000]
# m2v_list = [12.739276, 12.914400, 14.337075, 17.702659, 36.163392, 50.786536, 68.258777, 84.662921, 100.802565]
# lle_list = [3.394701, 3.833216, 9.272332, 14.402140, 69.345096, 146.910744, 242.717252, 349.964908, 476.670194]
# le_list = [0.842819, 1.671205, 6.907714, 12.672460, 55.757245, 119.352503, 195.739323, 297.003762, 403.991168]
# umap_list = [9.252944, 16.976110, 47.224855, 86.782111, 412.319919, 810.250494, 1252.856751, 1606.028516, 2058.503160]
# Avec_list = [11.113298, 12.371767, 18.519851, 20.413084, 26.948365, 28.422595, 32.224374, 37.975404, 42.950793]
# #isomap_list = [19.327131, 21.311042, 27.121375, 35.414925, 89.508136, 146.786021, 205.591825, 264.972651, 317.100561]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.plot(x1_list, lle_list, color='green', label='LLE', marker='v')
# plt.plot(x1_list, le_list, color='cyan', label='LE', marker='o')
# #plt.plot(x1_list, isomap_list, color='blue', label='Isomap', marker='+')
# plt.plot(x1_list, umap_list, color='gray', label='UMAP', marker='>')
# plt.plot(x1_list, m2v_list, color='red', label='Vec2vec', marker='x')
# plt.plot(x1_list, Avec_list, color='blue', label='AVec2vec', marker='d')
#
# # plt.xticks([1, 2, 3, 4, 5],[r'$100$', r'$1000$', r'$10000$', r'$100000$', r'$200000$'])
# plt.legend(fontsize=14)
# plt.xlabel('Dimensionality of data',fontsize=14)
# plt.ylabel('Running Time (s)',fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.tight_layout()
# plt.grid()
# plt.show()



# #matrix2vec with the change of dimensions d for dataset MNIST
# d_list = [1] +[n for n in range(16,513,16)]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [0.9394,
#             0.9370,0.9380,0.9386,0.9384,0.9394,0.9388,0.9398,0.9396,0.9398,0.9378,0.9414,0.9374,0.9386,0.9406,0.9382,0.940,0.9398,0.9384,
#             0.9384, 0.9370,0.9380,0.9382,0.9392,0.9392,0.9398,0.9386,0.9386,0.9382,0.9376,0.9392,0.9378,0.9376]
#
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.plot(d_list, m2v_list, color='green', label='LLE', marker='.')
#
#
# plt.xlabel('Embedding Dimensions')
# plt.ylim((0.9,0.96))
# plt.ylabel('Accuracy')
# plt.grid()
# plt.legend()  # 显示图例
# plt.show()



# #matrix2vec with the change of dimensions d for dataset COIL20   numberofwalks
# d_list = [n for n in range(2,33,2)]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [0.9340,0.9424,0.9389,0.9361,0.9313,0.9326,0.9375,0.9361,0.9375,0.9340,0.9319,0.9354,0.9382,0.9389,0.9389,0.9319]
#
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green', marker='.')
#
#
# plt.xlabel('Number of walks',fontsize=12)
# plt.ylim((0.92,0.945))
# plt.ylabel('Accuracy',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid()
# plt.tight_layout()
# plt.legend()  # 显示图例
# plt.show()


# #matrix2vec with the change of dimensions d for dataset COIL20.   lengthofwalks
# d_list = [n for n in range(2,33,2)]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [0.3819,0.8771,0.9264,0.9257,0.9333,0.9368,0.9340,0.9306,0.9326,0.9410,0.9354,0.9326,0.9389,0.9347,0.9368,0.9375]
#
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green', marker='.')
#
# plt.xlabel('Length of a walk',fontsize=12)
# # plt.ylim((0.9,0.96))
# plt.ylabel('Accuracy',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid()
# plt.tight_layout()
# plt.legend()  # 显示图例
# plt.show()


# #matrix2vec with the change of dimensions d for dataset COIL20.   dimensions
# d_list = [1,3,5,7,9,11]+[n for n in range(16,500,16)]
# m2v_list = [0.0590,0.6854,0.8944,0.9437,0.9347,0.9354]+[0.9382,0.9368,0.9382,0.9292,0.9361,0.9403,0.9326,0.9361,0.9417,0.9319,0.9361,0.9368,0.9396,0.9361,0.9361,0.9361,0.9278,0.9361,0.9354,0.9326,0.9375,0.9361,0.9389,0.9361,0.9313,0.9403,0.9292,0.9375,0.9333,0.9250,0.9299]
# # d_list = [1,3,5,7,9,11]+[n for n in range(16,161,16)]
# # m2v_list = [0.0590,0.6854,0.8944,0.9437,0.9347,0.9354]+[0.9382,0.9368,0.9382,0.9292,0.9361,0.9403,0.9326,0.9361,0.9417,0.9319]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green', marker='.')
#
# plt.xlabel('Output dimensionality',fontsize=12)
# # plt.ylim((0.915,0.945))
# plt.ylabel('Accuracy',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid()
# plt.tight_layout()
# plt.legend()  # 显示图例
# plt.show()

# # Image classification running time on four image dataset. target dimensionality
# x1_list = [1,3,5,7,9,11,13,15,16,32,48,64,80,96,112,128]
# cifar_list = [0.1148, 0.2010, 0.2482, 0.2832, 0.2934, 0.3046, 0.2998, 0.3094, 0.3028, 0.3064, 0.3132, 0.3198, 0.3224, 0.3278, 0.3178, 0.3190]
# MNIST_list = [0.1364, 0.7000, 0.9142, 0.9342, 0.9420, 0.9408, 0.9420, 0.9434, 0.9418, 0.9410, 0.9414, 0.9434, 0.9390, 0.9428, 0.9396, 0.9408]
# svhn_list = [0.1610, 0.1770, 0.2350, 0.2820, 0.3236, 0.3360, 0.3700, 0.3846, 0.3886, 0.4172, 0.4130, 0.4274, 0.4166, 0.4270, 0.4222, 0.4218]
# coil20_list = [0.0833, 0.7847, 0.8792, 0.8917, 0.8701, 0.8743, 0.8743, 0.8868, 0.8792, 0.8785, 0.8729, 0.8799, 0.8799, 0.8847, 0.8736, 0.8847]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.plot(x1_list, cifar_list, color='green', label='CIFAR-10', marker='v')
# plt.plot(x1_list, MNIST_list, color='cyan', label='MNIST', marker='o')
# plt.plot(x1_list, svhn_list, color='blue', label='SVHN', marker='+')
# plt.plot(x1_list, coil20_list, color='red', label='COIL20', marker='x')
#
# plt.legend(fontsize=14,bbox_to_anchor=(0.96,0.8))
# plt.xlabel('Target Dimensionality',fontsize=14)
# plt.ylabel('Accuracy',fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.tight_layout()
# plt.grid()
# plt.show()




# # Image classification running time on four image dataset. topk in building similarity graph
# x1_list = [1,2,3,4,5,6,7,8,9]
# cifar_list = [0.2873, 0.3273, 0.3317, 0.3331, 0.3317, 0.3344, 0.3352, 0.3343, 0.3345]
# MNIST_list = [0.8781, 0.9394, 0.9456, 0.9456, 0.9462, 0.9460, 0.9440, 0.9448, 0.9410]
# svhn_list = [0.3390, 0.4012, 0.4356, 0.4366, 0.4392, 0.4438, 0.4334, 0.4344, 0.4258]
# # pca_list = [0.795412, 2.650757, 8.661204, 15.750142, 22.746649]
# coil20_list = [0.2056, 0.9382, 0.9410, 0.9097, 0.9000, 0.8972, 0.8944, 0.8944, 0.8833]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.plot(x1_list, cifar_list, color='green', label='CIFAR-10', marker='v')
# plt.plot(x1_list, MNIST_list, color='cyan', label='MNIST', marker='o')
# plt.plot(x1_list, svhn_list, color='blue', label='SVHN', marker='+')
# plt.plot(x1_list, coil20_list, color='red', label='COIL20', marker='x')
#
# plt.legend(fontsize=14,bbox_to_anchor=(0.96,0.8))
# plt.xlabel('Number of topk neighbours',fontsize=14)
# plt.ylabel('Accuracy',fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.tight_layout()
# plt.grid()
# plt.show()




# #matrix2vec with the change of dimensions d for dataset COIL20.   number of contexts
# d_list = [n for n in range(2,33,2)]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [0.7917,0.9167,0.9368,0.9292,0.9389,0.9347,0.9354,0.9326,0.9333,0.9382,0.9340,0.9326,0.9417,0.9389,0.9424,0.9410]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green', marker='.')
#
#
# plt.xlabel('Sliding window of the context',fontsize=12)
# # plt.ylim((0.92,0.95))
# plt.ylabel('Accuracy',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid()
# plt.tight_layout()
# foo_fig = plt.gcf()
# foo_fig.savefig('foo.eps', format='eps', dpi=1000)
# plt.show()




# #matrix2vec with the change of dimensions d for dataset COIL20.   number of iters
# d_list = [n for n in range(5,51,5)]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [0.9361,0.9292,0.9306,0.9361,0.9375,0.9319,0.9375,0.9382,0.9292,0.9306]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green', marker='.')
#
#
# plt.xlabel('Number of iterations',fontsize=12)
# plt.ylim((0.91,0.94))
# plt.ylabel('Accuracy',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid()
# plt.tight_layout()
# plt.legend()  # 显示图例
# plt.show()



# #matrix2vec with the change of dimensions d for dataset COIL20.   number of top_k neighbours
# d_list = [n for n in range(1,10,1)]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [0.2056,0.9382,0.9410,0.9097,0.9000,0.8972,0.8944,0.8944,0.8833]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green', marker='.')
#
# plt.xlabel('Number of topk neighbours',fontsize=12)
# # plt.ylim((0.92,0.95))
# plt.ylabel('Accuracy',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# plt.grid()
# plt.tight_layout()
# plt.legend()  # 显示图例
# plt.show()



# #matrix2vec with the change of dimensions d for dataset COIL20.   Memories
# d_list = [1000.0,10000.0,40000.0,100000]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [170.79,1365.74,14231.11,47014.00]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# # plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green', marker='.')
#
# plt.xlabel('Number of Observations',fontsize=12)
# plt.ylim((100,150000))
# plt.ylabel('Memory (MB)',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xscale('log')
# plt.yscale('log')
#
# plt.grid()
# plt.tight_layout()
# plt.legend()  # 显示图例
# plt.show()



#matrix2vec with the change of dimensions d for dataset COIL20.   Computation time
# d_list = [100,1000.0,10000.0,100000]
# # d_list = [n for n in range(16,513,16)]
# m2v_list = [2.46,29.28,632.63,44798.21]
# nn_list=[0.534,4.97,394.69,38689.61]
# mn_list=np.array(m2v_list)-np.array(nn_list)
# print(mn_list)
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# # plt.figure(figsize=(4,3.2))
# plt.plot(d_list, m2v_list, color='green',label='Matrix2vec', marker='.')
# plt.plot(d_list, nn_list, color='cyan', label='Adjacency Graph', marker='v')
# plt.plot(d_list, mn_list, color='red', label='Random Walk and Skip-gram', marker='+')
#
# plt.xlabel('Number of Observations',fontsize=12)
# plt.ylim((0.1,100000))
# plt.ylabel('Memory (MB)',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xscale('log')
# plt.yscale('log')
#
# plt.grid()
# plt.tight_layout()
# plt.legend(fontsize=12)  # 显示图例
# plt.show()


# #matrix2vec with the change of dimensions d for dataset SVHN.   Computation time
# d_list = [10,100,1000.0,10000.0,50000.0]
# # d_list = [n for n in range(16,513,16)]
# lle_list = [0.539229,1.426253,32.099182,485.458168,9293.602906]
# le_list=[0.163206,1.430909,20.141829,487.311030,9169.879578]
# pca_list=[0.160736,1.421562,18.175213,484.820335,9162.752003]
# mds_list=[0.170716,1.417874,18.164910,483.994209,9124.671877]
# isomap_list=[0.172467,1.422357,18.244332,484.077603,9110.799602]
# vec2vec_list=[1.007664,2.370926,16.797099,169.167664,2318.417830]
# #开始画图
# # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
# # plt.title('Movie')
# plt.figure(figsize=(5,4))
# plt.plot(d_list, lle_list, color='green',label='LLE', marker='.')
# plt.plot(d_list, le_list, color='cyan', label='LE', marker='v')
# plt.plot(d_list, pca_list, color='blue', label='PCA', marker='+')
# plt.plot(d_list, mds_list, color='magenta',label='MDS', marker='x')
# plt.plot(d_list, isomap_list, color='deepskyblue',label='Isomap', marker='*')
# plt.plot(d_list, vec2vec_list, color='red',label='Vec2vec', marker='o')
#
#
# plt.xlabel('Number of Observations',fontsize=12)
# plt.ylim((0.1,10000))
# plt.ylabel('Computational Time (s)',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xscale('log')
# plt.yscale('log')
#
# plt.grid()
# plt.tight_layout()
# plt.legend(fontsize=12)  # 显示图例
# plt.show()




# index=np.arange(5)
# value=[5,7,3,4,6]
# std1=[0.8,1.0,0.4,0.9,1.3]
# plt.title('A bar Chart with errorbars')
# plt.bar(index,value,yerr=std1,error_kw={'ecolor':'0.3', 'capsize':10},alpha=0.7,label='First')
# plt.xticks(index,['A','B','C','D','E'])
# plt.legend(loc=2)
# plt.show()

# x = np.linspace(0, 10, 50)
# dy = np.random.normal(0, 1, 50)
# y = np.sin(x) + dy*np.random.randn(50)
# plt.subplot(211)
# plt.errorbar(x, y, yerr=dy, fmt='.k')
# plt.grid(True)
# plt.subplot(212)
# plt.errorbar(x, y, yerr=0.8, fmt='o')
# plt.grid(True)
# plt.show()

# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 35, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
# bottem_women_means=[24, 31, 33, 19, 24]
# men_std = [2, 3, 4, 1, 2]
# women_std = [3, 5, 2, 3, 3]
# width = 0.35       # the width of the bars: can also be len(x) sequence
#
# fig, ax = plt.subplots()
#
# ax.bar(labels, men_means, width, yerr=men_std, label='Men')
# ax.bar(labels, women_means, width, yerr=women_std, bottom=bottem_women_means,
#        label='Women')
#
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.legend()
#
# plt.show()




# example data
# x = np.arange(0.1, 4, 0.5)
# y = np.exp(-x)
#
# # example variable error bar values
# yerr = 0.1 + 0.2*np.sqrt(x)
# xerr = 0.1 + yerr
#
# # First illustrate basic pyplot interface, using defaults where possible.
# plt.figure()
# plt.errorbar(x, y, xerr=0.2, yerr=0.4,xlolims=True,xuplims=True)
# plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")

# # Now switch to a more OO interface to exercise more features.
# fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
# ax = axs[0,0]
# ax.errorbar(x, y, yerr=yerr, fmt='o')
# ax.set_title('Vert. symmetric')
#
# # With 4 subplots, reduce the number of axis ticks to avoid crowding.
# ax.locator_params(nbins=4)
#
# ax = axs[0,1]
# ax.errorbar(x, y, xerr=xerr, fmt='o',xlolims=True,xuplims=True)
# ax.set_title('Hor. symmetric')
#
# ax = axs[1,0]
# ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
# ax.set_title('H, V asymmetric')
#
# ax = axs[1,1]
# ax.set_yscale('log')
# # Here we have to be careful to keep all y values positive:
# ylower = np.maximum(1e-2, y - yerr)
# yerr_lower = y - ylower
#
# ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
#             fmt='-o', ecolor='g', capthick=2)
# ax.set_title('Mixed sym., log y')

# fig.suptitle('Variable errorbars')

# plt.show()


# x=np.linspace(0.1,0.5,10)#生成[0.1,0.5]等间隔的十个数据
# y=np.exp(x)
#
# error=0.05+0.15*x#误差范围函数
#
# error_range=[error*0.3,error]#下置信度和上置信度
#
# plt.errorbar(x,y,yerr=error_range,fmt='o:',ecolor='hotpink',elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
#
# plt.xlim(0.05,0.55)#设置x轴显示范围区间
# plt.show()



# n_groups = 5
#
# means_men = (20, 35, 30, 35, 27)
# std_men = (2, 3, 4, 1, 2)
#
# means_women = (25, 32, 34, 20, 25)
# std_women = (3, 5, 2, 3, 3)
#
# fig, ax = plt.subplots()
#
# index = np.arange(n_groups)
# bar_width = 0.35
#
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
#
# rects1 = ax.bar(index, means_men, bar_width,
#                 alpha=opacity, color='b',
#                 yerr=std_men, error_kw=error_config,
#                 label='Men')
#
# rects2 = ax.bar(index + bar_width, means_women, bar_width,
#                 alpha=opacity, color='r',
#                 yerr=std_women, error_kw=error_config,
#                 label='Women')
#
# ax.set_xlabel('Group')
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
# ax.legend()
#
# fig.tight_layout()
# plt.show()



# Text and Image Classification of Dataset "MNIST"
df = pd.DataFrame({"Method": ['PCA','MDS','Isomap','LE','LLE','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.9354,0.8588,0.9156,0.9332,0.9290,0.9418,0.9462,0.9250],
                   'yerr': [0.0141,0.0190,0.0132,0.0151,0.0217,0.0148,0.0182,0.0170]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
# +xlab("Method")
# +xlim(0,7)
+ylim(0.83,0.98)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text = element_text(size=12),
       axis_text_x = element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x = element_blank(),
       aspect_ratio =1.0,
       dpi=100,
       figure_size=(5,3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_MNIST.eps")



# Text and Image Classification of Dataset "COIL20"
df = pd.DataFrame({"Method": ['PCA','LLE','LE','MDS','Isomap','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.9222, 0.8993, 0.8965, 0.8812, 0.9028, 0.9576, 0.9410, 0.9056],
                   'yerr': [0.0219, 0.0341, 0.0284, 0.0107, 0.0573, 0.0217, 0.0168, 0.0307]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
+xlab("Method")
# +xlim(0,7)
+ylim(0.83,1.0)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text=element_text(size=12),
       axis_text_x=element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x=element_blank(),
       aspect_ratio=1.0,
       dpi=100,
       figure_size=(5, 3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_coil20.eps")


# Text and Image Classification of Dataset "CIFAR10"
df = pd.DataFrame({"Method": ['PCA','LLE','LE','MDS','Isomap','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.3134, 0.3138, 0.2824, 0.2568, 0.2802, 0.3274, 0.3358, 0.3208],
                   'yerr': [0.0311, 0.0410, 0.0116, 0.0029, 0.0168, 0.0137, 0.0059, 0.0296]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
+xlab("Method")
# +xlim(0,7)
+ylim(0.25,0.36)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text=element_text(size=12),
       axis_text_x=element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x=element_blank(),
       aspect_ratio=1.0,
       dpi=100,
       figure_size=(5, 3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_cifar10.eps")




# Text and Image Classification of Dataset "SVHN"
df = pd.DataFrame({"Method": ['PCA','LLE','LE','MDS','Isomap','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.3292, 0.308, 0.2356, 0.2182, 0.2288, 0.3924, 0.4450, 0.4154],
                   'yerr': [0.0359, 0.0452, 0.0106, 0.023, 0.0236, 0.0239, 0.0342, 0.0262]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
+xlab("Method")
# +xlim(0,7)
+ylim(0.18, 0.50)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text=element_text(size=12),
       axis_text_x=element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x=element_blank(),
       aspect_ratio=1.0,
       dpi=100,
       figure_size=(5, 3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_svhn.eps")



# Text and Image Classification of Dataset "GoogleSnippets"
df = pd.DataFrame({"Method": ['PCA','LLE','LE','MDS','Isomap','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.9210, 0.7970, 0.7675, 0.8565, 0.8735, 0.8768, 0.9135, 0.8202],
                   'yerr': [0.0681, 0.1127, 0.0402, 0.1504, 0.1552, 0.0259, 0.0633, 0.0464]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
+xlab("Method")
# +xlim(0,7)
+ylim(0.65, 1.05)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text=element_text(size=12),
       axis_text_x=element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x=element_blank(),
       aspect_ratio=1.0,
       dpi=100,
       figure_size=(5, 3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_GoogleSnippets.eps")



# Text and Image Classification of Dataset "20News"
df = pd.DataFrame({"Method": ['PCA','LLE','LE','MDS','Isomap','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.4006, 0.4258, 0.4022, 0.0862, 0.4152, 0.4634, 0.5452, 0.4682],
                   'yerr': [0.0248, 0.0277, 0.0287, 0.0208, 0.0367, 0.0335, 0.0128, 0.0330]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
+xlab("Method")
# +xlim(0,7)
+ylim(0.05, 0.60)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text=element_text(size=12),
       axis_text_x=element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x=element_blank(),
       aspect_ratio=1.0,
       dpi=100,
       figure_size=(5, 3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_20News.eps")



# Text and Image Classification of Dataset "20NewsLong"
df = pd.DataFrame({"Method": ['PCA','LLE','LE','MDS','Isomap','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.4864, 0.498, 0.4805, 0.1575, 0.3613, 0.7925, 0.771, 0.6185],
                   'yerr': [0.06, 0.0199, 0.038, 0.0197, 0.0418, 0.0111, 0.0194, 0.0214]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
+xlab("Method")
# +xlim(0,7)
+ylim(0.10, 0.85)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text=element_text(size=12),
       axis_text_x=element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x=element_blank(),
       aspect_ratio=1.0,
       dpi=100,
       figure_size=(5, 3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_20NewsLong.eps")




# Text and Image Classification of Dataset "Movie"
df = pd.DataFrame({"Method": ['PCA','LLE','LE','MDS','Isomap','UMAP','Vec2vec','AVec2vec'],
                   'Accu': [0.6875, 0.665, 0.6555, 0.5345, 0.6415, 0.6745, 0.6875, 0.6590],
                   'yerr': [0.038, 0.0254, 0.0325, 0.0312, 0.0266, 0.0372, 0.038, 0.0263]})

jitter_plot=(ggplot(df, aes(x ='Method', y = 'Accu', ymin = 'Accu-yerr', ymax = 'Accu+yerr'))
+geom_line(size=1,show_legend=True)
+geom_errorbar(colour="black", width=0.2,size=1,show_legend=True)
+geom_point(aes(fill = 'Method'),shape='o',size=4,stroke=0.75,show_legend=True)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='hls')
+ylab("Accuracy")
+xlab("Method")
# +xlim(0,7)
+ylim(0.50, 0.75)
+scale_x_discrete(limits=('PCA','MDS','Isomap','LE','LLE','AVec2vec','UMAP','Vec2vec'))
+theme(#legend_position= (0.52,0.80),
       # plot_margin= None,
       legend_position= 'none',
       # plot_background = element_rect(color = "red",size = 3),
       # plot_margin = unit(t = 0, r = 0, b = 0, l = 0),  # 左边边缘距离
       legend_title=element_blank(),
       # legend_title_align = 'center',
       legend_direction = "horizontal",
       legend_text=element_text(size=12),
       axis_text_x=element_text(angle=45, hjust=1, size=12),
       axis_text_y=element_text(size=12),
       axis_title_x=element_blank(),
       aspect_ratio=1.0,
       dpi=100,
       figure_size=(5, 3),
       title=element_text(size=12)))

print(jitter_plot)
jitter_plot.save("D:/NSFC/project/TKDE2020/latex/fig/classification_movie.eps")