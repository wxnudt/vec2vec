#!/usr/bin/env python
# -*- coding: utf-8 -*-

## For Testing Matrix2vec on dataset MNIST
## PCA, Kernel PCA, ISOMAP,  NMDS, LLE, LE

# import tensorflow as ts
import logging
import os.path
import sys
import multiprocessing
import numpy as np
import argparse
import scipy.io
import datetime
import matrix2vec


from sklearn import datasets as ds
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
# from keras.datasets import mnist
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from sklearn.model_selection  import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
from sklearn.preprocessing import scale
#import cPickle as pickle
import pickle
from scipy import misc
import  matplotlib.image as mpimg

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return np.array(dict['data']), np.array(dict['labels'])

def load_data(path):
    x_train, y_train = ds.load_svmlight_file(path)
    x_train.todense()
    return x_train,y_train



def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    # train, val, test = pickle.load(f)
    train, val, test = pickle.load(f, encoding='bytes')
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

def resizeSVHDShape(matrix):
    svhd = np.zeros((5000,3072))
    [rows, cols] = svhd.shape
    for r in range(rows):
        for c in range(cols):
            svhd[r][c]=matrix[(c%1024)/32][(c%1024)%32][c/1024][r]
    return svhd


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

    # CoilData = scipy.io.loadmat("D:/NSFC/project/data/coil20/COIL20.mat")  # Loading coil20.mat
    # coil_x = CoilData['X']
    # coil_y = CoilData['Y']
    # x_train = coil_x
    # y_train = []
    # for item in coil_y:
    #     y_train.append(item[0])
    # print("Load the COIL20 dataset finished...")

    # SVHDData = scipy.io.loadmat("D:/NSFC/project/data/SVHN/train_32x32.mat")  # Loading SVHN
    # svhd_x = SVHDData['X']
    # x_train = resizeSVHDShape(svhd_x)
    # svhd_y = SVHDData['y']
    # y_train = []
    # for item in svhd_y:
    #     y_train.append(item[0])
    # print("Load dataset finished...")


    #data,label =load_data("D:/NSFC/project/data/movie/train.bow")
    # data, label = load_data(args.input)
    x_train, y_train, x_test, y_test = read_data("D:/NSFC/project/data/MNIST/origData/mnistpklgz/mnist.pkl.gz")
    print("Load dataset MNIST finished...")

    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/20NewLong/train.bow')
    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/20New/20news.train.bow')
    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/movie/train.bow')
    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/GoogleSnippets/traintest.txt.bow')
    # x_train2, y_train2 = ds.load_svmlight_file('/Users/renxl/Desktop/todo/re4/data-clear/GoogleSnippets/traintest.txt.bow')
    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/stackoverflow/title_label.txt.bow')

    # x_train, x_te, y_train, y_te = train_test_split(x_train2, y_train2, train_size=500)


    # x_train, y_train = unpickle("D:/NSFC/project/data/cifar10/cifar-10-batches-py/data_batch_1")

    # print("Load dataset 20News finished...")

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # x_train=x_train[0:5000, :]
    # y_train = y_train[0:5000]

    # print(x_train2.shape)
    # print(x_train2)
    # x_train = x_train2.toarray()
    # y_train = y_train2


    # x_train=MinMaxScaler().fit_transform(x_train)
    x_train=x_train[0:5000, :]
    y_train = y_train[0:5000]

    print(x_train.shape)
    print(x_train)

    models = []
    emb_size=64
    num_neighbors=32

    for emb_size in (32,64):
        print("********************* emb_size="+str(emb_size)+" ***************")

        models=[]
        models.append(LocallyLinearEmbedding(n_neighbors=num_neighbors,n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
        models.append(SpectralEmbedding(n_neighbors=num_neighbors,n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
        models.append(PCA(n_components=emb_size))
        models.append(MDS(n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
        models.append(Isomap(n_neighbors=num_neighbors, n_components=emb_size, n_jobs=multiprocessing.cpu_count()))
        models.append('matrix2vec')

        model_names = ['lle', 'le', 'pca', 'MDS', 'ISOMAP', 'matrix2vec']  # names corresponding to model


        for index, embedding in enumerate(models):
            print('Start running model '+model_names[index]+"...")
            start = datetime.datetime.now()
            X_transformed= np.zeros((x_train.shape[0],emb_size))
            if(index<=4):
                # X_transformed = embedding.fit_transform(x_train)
                X_transformed = embedding.fit_transform(x_train)

            else:
                X_transformed=matrix2vec.matrix2vec(x_train,emb_size,topk=5,num_iter=10)
            end = datetime.datetime.now()

            #scale
            X_transformed=scale(X_transformed)

            
            print('Model '+model_names[index]+' Finished in '+str(end-start)+" s.")

            #Using KNN classifier to test the result with cross_validation
            knn = KNeighborsClassifier()
            param = {"n_neighbors": [1, 3, 5, 7, 11]}  # 构造一些参数的值进行搜索 (字典类型，可以有多个参数)
            gc = GridSearchCV(knn, param_grid=param, cv=4)
            gc.fit(X_transformed, y_train)
            knn = gc.best_estimator_
            print("The best parameter: n_neighbors=" + str(knn.n_neighbors))
            scores = cross_val_score(knn, X_transformed, y_train, cv=4)
            print("交叉验证Accuracy： ", scores)
            print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
