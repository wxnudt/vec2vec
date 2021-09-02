#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing
import numpy as np
import argparse
import matrix2vec
import datetime
import scipy.io

from sklearn import datasets
from sklearn.cluster import SpectralClustering,KMeans
from sklearn import metrics
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.manifold import LocallyLinearEmbedding
# from keras.datasets import mnist
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
# import cPickle as pickle
import pickle
from sklearn.cluster import SpectralClustering


def load_data(path):
    x_train, y_train = datasets.load_svmlight_file(path)
    x_train.todense()
    return x_train,y_train


def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    # train, val, test = pickle.load(f)
    train, val, test = pickle.load(f, encoding='bytes')
    f.close()
    # print(train)
    train_x = train[0]
    train_y = train[1]
    #
    # print(train_x)
    # print(train_y)


    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

def resizeSVHDShape(matrix):
    svhd = np.zeros((5000,3072))
    [rows, cols] = svhd.shape
    for r in range(rows):
        for c in range(cols):
            svhd[r][c]=matrix[int((c%1024)/32)][(c%1024)%32][int(c/1024)][r]
    return svhd



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


if __name__ == "__main__":
    args = parse_args()

    # CoilData = scipy.io.loadmat("D:/NSFC/project/data/coil20/COIL20.mat")  # Loading coil20.mat
    # coil_x = CoilData['X']
    # coil_y = CoilData['Y']
    # x_train = coil_x
    # y_train = []
    # for item in coil_y:
    #     y_train.append(item[0])
    # print("Load dataset finished...")

    # SVHDData = scipy.io.loadmat("D:/NSFC/project/data/SVHN/train_32x32.mat")  # Loading coil20.mat
    # svhd_x = SVHDData['X']
    # x_train = resizeSVHDShape(svhd_x)
    # svhd_y = SVHDData['y']
    # y_train = []
    # for item in svhd_y:
    #     y_train.append(item[0])
    # print("Load dataset SVHN finished...")
    #

    # x_train, y_train = unpickle("D:/NSFC/project/data/cifar10/cifar-10-batches-py/data_batch_1")

    # data,label =load_data("D:/NSFC/project/data/movie/train.bow")
    # data, label = load_data(args.input)
    # x_train, y_train, x_test, y_test = read_data("D:/NSFC/project/data/MNIST/origData/mnistpklgz/mnist.pkl.gz")

    # x_train, y_train = ds.load_svmlight_file('D:/NSFC/project/data/20New/20news.train.bow')
    # x_train, y_train = ds.load_svmlight_file('D:/NSFC/project/data/20NewLong/train.bow')

    x_train, y_train = ds.load_svmlight_file('D:/NSFC/project/data/movie/train.bow')
    # x_train, y_train = ds.load_svmlight_file('D:/NSFC/project/data/GoogleSnippets/traintest.txt.bow')

    # x_train, x_te, y_train, y_te = train_test_split(x_train2, y_train2, train_size=500)
    # x_train = x_train.toarray()
    #     # y_train = y_train2

    print("Load dataset finished...")

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    x_train=x_train[0:2000, :].toarray()
    y_train = y_train[0:2000]

    # print(x_train)
    print(x_train.shape)
    # print(y_train.shape)


    # x_train=MinMaxScaler().fit_transform(x_train)

    for num_walks in range(5, 55, 5):
        print("*************** num_walks="+str(num_walks)+" *****************")
        models = []
        emb_size = 32
        # models.append(LocallyLinearEmbedding(n_neighbors=num_neighbors,n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
        # models.append(SpectralEmbedding(n_neighbors=num_neighbors,n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
        # models.append(PCA(n_components=emb_size))
        # models.append(MDS(n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
        # models.append(Isomap(n_neighbors=num_neighbors, n_components=emb_size, n_jobs=multiprocessing.cpu_count()))
        models.append('matrix2vec')

        # model_names=['le','pca','MDS','ISOMAP','matrix2vec'] # names corresponding to models
        model_names = ['lle', 'le', 'pca', 'MDS', 'ISOMAP', 'matrix2vec']  # names corresponding to models
        # model_names = ['lle']  # names corresponding to models

        # model_names=['MDS','ISOMAP'] # names corresponding to models

        for index, embedding in enumerate(models):
            print('Start running model ' + model_names[index] + "...")
            start = datetime.datetime.now()
            X_transformed = np.zeros((x_train.shape[0], emb_size))
            if(index<=-1):
                X_transformed = embedding.fit_transform(x_train)
                # continue
                # X_transformed = embedding.fit_transform(x_train.toarray())
            else:
                X_transformed = matrix2vec.matrix2vec(x_train, emb_size, num_walks=num_walks,
                                                      walk_length=30, num_iter=10, topk=10)
            end = datetime.datetime.now()

            outputpath = "D:/NSFC/project/data/AppVec2vec/clustering/MNIST-clusterResults" + ".SpeClu"
            out = []
            for gamma in (1e-8, 1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1):
            # for gamma in (1e-8,1e-7):
                y_pred = SpectralClustering(n_clusters=2, gamma=gamma,n_jobs=multiprocessing.cpu_count()-1).fit_predict(X_transformed)
                # y_pred = KMeans(n_clusters=2, n_jobs=multiprocessing.cpu_count() - 1).fit_predict(X_transformed)
                print("Adjusted Rand Index with gamma="+str(gamma)+" score:"+str(metrics.adjusted_rand_score(y_train,  y_pred)))
                out.append(str(model_names[index])+" Adjusted Rand Index with gamma="+str(gamma)+" score:"+str(metrics.adjusted_rand_score(y_train,  y_pred)))
        np.savetxt(outputpath, out, fmt="%s")

