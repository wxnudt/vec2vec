#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Running Matrix2vec on dataset MNIST

import logging
import os.path
import sys
import numpy as np
import multiprocessing
import argparse
import datetime
import operator
import scipy.io
import matrix2vec
import h5py

from sklearn import datasets
from sklearn.model_selection  import train_test_split, GridSearchCV, cross_val_score,cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle



def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y


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


if __name__ == "__main__":
    #args = parse_args()

    x_train, y_train, x_test, y_test=read_data("D:/NSFC/project/data/MNIST/origData/mnistpklgz/mnist.pkl.gz")
    x_train = x_train[0:1000, :]
    y_train = y_train[0:1000]

    emb_size=64
    num_neighbors=32

    # for top_k in range(1,11,1):
    start = datetime.datetime.now()
    X_transformed = np.zeros((x_train.shape[0], emb_size))
    X_transformed = matrix2vec.matrix2vec(x_train,emb_size,num_iter=it,topk=3)
    # X_transformed=MinMaxScaler().fit_transform(X_transformed)
    end = datetime.datetime.now()


    knn = KNeighborsClassifier()
    param = {"n_neighbors": [1,3, 5, 7, 11]}  # 构造一些参数的值进行搜索 (字典类型，可以有多个参数)
    gc = GridSearchCV(knn, param_grid=param, cv=4)
    gc.fit(X_transformed, y_train)
    knn = gc.best_estimator_
    # print("The best parameter: n_neighbors=" + str(knn.n_neighbors) )

    scores = cross_val_score(knn, X_transformed, y_train, cv=4)
    # print("交叉验证Accuracy： ", scores)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))