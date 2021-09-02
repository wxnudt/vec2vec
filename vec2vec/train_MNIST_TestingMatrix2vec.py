#!/usr/bin/env python
# -*- coding: utf-8 -*-

## For Testing Matrix2vec on dataset MNIST
## PCA, Kernel PCA, ISOMAP,  NMDS, LLE, LE

import logging
import os.path
import sys
import numpy as np
import multiprocessing
import argparse
import datetime
import operator
import scipy.io
# import matrix2vec_rxl
import matrix2vec
# import randommatrix2vec
import h5py

from functools import reduce
from sklearn import datasets
from sklearn import datasets as ds
from sklearn.manifold import LocallyLinearEmbedding
# from keras.datasets import mnist
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
# from sklearn.datasets import fetch_mldata
from itertools import chain
# import cPickle as pickle
import pickle as pickle
from scipy import misc
import matplotlib.image as mpimg
from sklearn.preprocessing import scale as scale_fun
# import vec2vec
# import quickmatrix2vec


def load_data(path):
    x_train, y_train = datasets.load_svmlight_file(path)
    x_train.todense()
    return x_train, y_train


# def load_mnist_dataset():
#     (x_train, x_train_label), (x_test, y_test) = mnist.load_data()
#     print(x_train_label)
#
#     x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
#     x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
#     x_train = x_train.reshape(x_train.shape[0], -1)
#     x_test = x_test.reshape(x_test.shape[0], -1)
#     print(x_train.shape)
#     print(x_test.shape)
#     return x_train, x_train_label, x_test, y_test

def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f, encoding='iso-8859-1')
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='iso-8859-1')
    return np.array(dict['data']), np.array(dict['labels'])


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run LLE, LE and PCA Algorithm.")
    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')
    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    return parser.parse_args()


k = 10
scale = 1.0


# covert image to sole vector
def img2vector(filename):
    imgVector = misc.imresize(mpimg.imread(filename), scale).flatten()
    return imgVector.astype(np.float)


# load image from diretion
def loadimage(dataSetDir):
    train_face = np.zeros((40 * k, int(112 * scale) * int(92 * scale)))  # image size:112*92
    train_face_number = np.zeros(40 * k).astype(np.int8)
    test_face = np.zeros((40 * (10 - k), int(112 * scale) * int(92 * scale)))
    test_face_number = np.zeros(40 * (10 - k)).astype(np.int8)
    for i in np.linspace(1, 40, 40).astype(np.int8):  # 40 sample people
        people_num = i
        for j in np.linspace(1, 10, 10).astype(np.int8):  # everyone has 10 different face
            if j <= k:
                filename = dataSetDir + '/s' + str(people_num) + '/' + str(j) + '.pgm'
                img = img2vector(filename)
                train_face[(i - 1) * k + (j - 1), :] = img
                train_face_number[(i - 1) * k + (j - 1)] = people_num
            else:
                filename = dataSetDir + '/s' + str(people_num) + '/' + str(j) + '.pgm'
                img = img2vector(filename)
                test_face[(i - 1) * (10 - k) + (j - k) - 1, :] = img
                test_face_number[(i - 1) * (10 - k) + (j - k) - 1] = people_num

    return train_face, train_face_number, test_face, test_face_number  # tuple


def resizeSVHDShape(matrix):
    svhd = np.zeros((5000, 3072))
    [rows, cols] = svhd.shape
    for r in range(rows):
        for c in range(cols):
            svhd[r][c] = matrix[int((c % 1024) / 32)][(c % 1024) % 32][int(c / 1024)][r]
    return svhd


if __name__ == "__main__":

    x_train2, y_train2 = ds.load_svmlight_file('F:/projects/vec2vec/data-clear-xlren/data-clear/movie/train.bow')

    x_train = x_train2.toarray()
    y_train = y_train2

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    x_train = x_train[0:2000, :]
    y_train = y_train[0:2000]


    models = []
    emb_size = 64
    num_neighbors = 16
    print(x_train.shape)

    for num_walks in range(5, 55, 5):
        print("************* The number of num_walks is : " + str(num_walks) + " *******************")
        # for top_k in range(1,11,1):
        start = datetime.datetime.now()
        # x_train = scale_fun(x_train)
        X_transformed = np.zeros((x_train.shape[0], emb_size))
        # X_transformed = quickmatrix2vec.quickmatrix2vec(x_train,emb_size,num_iter=it,topk=5)
        X_transformed = matrix2vec.matrix2vec(x_train, emb_size, num_walks=num_walks,
                                              walk_length=30, num_iter=10, topk=10)

        end = datetime.datetime.now()

        # scale
        X_transformed = scale_fun(X_transformed)

        print('Model Matrix2vec Finished in ' + str(end - start) + " s.")
        # print('Model Matrix2vec with walk_length=' + str(wl) + ' Finished in ' + str(
        #     end - start) + " s.")

        # Using KNN classifier to test the result with cross_validation
        x_tr, x_te, y_tr, y_te = train_test_split(X_transformed, y_train, test_size=0.25)
        knn = KNeighborsClassifier()
        param = {"n_neighbors": [1, 3, 5, 7, 11]}  # 构造一些参数的值进行搜索 (字典类型，可以有多个参数)
        gc = GridSearchCV(knn, param_grid=param, cv=4)
        gc.fit(X_transformed, y_train)
        knn = gc.best_estimator_
        scores = cross_val_score(knn, X_transformed, y_train, cv=4)
        print("交叉验证Accuracy： ", scores)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
