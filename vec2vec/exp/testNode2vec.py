#!/usr/bin/env python
# -*- coding: utf-8 -*-

## For Testing Matrix2vec on dataset MNIST
## PCA, Kernel PCA, ISOMAP,  NMDS, LLE, LE

import logging
import os.path
import sys
import multiprocessing
import numpy as np
import argparse
import scipy.io
import datetime
import vec2vec.matrix2vec
import operator
import networkx as nx


from functools import reduce
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
# from keras.datasets import mnist
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from sklearn.model_selection  import train_test_split, GridSearchCV, cross_val_score,cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
from itertools import chain
from vec2vec.node2vec_master import node2vec
from gensim.models import Word2Vec
import numpy


def load_data(path):
    x_train, y_train = datasets.load_svmlight_file(path)
    x_train.todense()
    return x_train,y_train

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

    # x_train=x_train[0:100, :]
    # y_train = y_train[0:100]

    # data,label =load_data("D:/NSFC/project/data/movie/train.bow")
    # data, label = load_data(args.input)
    # x_train, y_train, x_test, y_test=load_mnist_dataset()


    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)


    # x_train=x_train[0:100, :]
    # y_train = y_train[0:100]
    # x_test=x_test[0:100,:]
    # y_test=y_test[0:100]
    # x_train = MinMaxScaler().fit_transform(x_train)
    # print(y_train)

    y_train=[2,2,2,2,1,2,1,2,2,1,2,1,2,2,2,1,1,2,1,2,2,1,2,2,1,1,1,1,2,2,2,1,2,1,2,2,2,1,2,2,1,1,1,1,2,1,1,2,2,1,1,1,2,2,2,2,2,2,2,1,2,1,1,1,2,1,2,1,2,2,1,1,1,1,2,1,1,1,2,2,2,2,2,2,2,2,1,1,2,2,1,1,1,2,1,1,1,1,2,1]
    models = []
    emb_size=128
    num_neighbors=32

    # X_transformed= np.zeros((x_train.shape[0],emb_size))
    # if(index<=4):
    #     X_transformed = embedding.fit_transform(x_train)
    # else:
    # for p_value in [1,2,3,4,5]:
    #     for q_value in [1,2,3,4,5]:

#    G = nx.read_edgelist("D:/NSFC/project/data/movie/train-100.bow.top800-test.net", nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    G = nx.read_edgelist("/Users/renxl/Desktop/todo/re4/data-clear/movie/train.bow", nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

    G = G.to_undirected()
    print("Load data finished...")
    G = node2vec.Graph(G, False,1,1)
    G.preprocess_transition_probs()
    print("11")
    walks = G.simulate_walks(10, 80)
    print("22")
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=emb_size, window=10, min_count=0, sg=1, workers=8, iter=20)
    print("33")
    output = '/Users/renxl/Desktop/todo/re4/data-clear/output/output_emb.m2v'
    model.wv.save_word2vec_format(output)

    X_transformed = numpy.ndarray((len(model.wv.index2word), emb_size))
    for index, word in enumerate(model.wv.index2word):
        X_transformed[int(word)-1] = numpy.array(model.wv.vectors[index])

    print("44")

    svc=svm.SVC(kernel='linear')
    print("The best parameter: Kernel="+str(svc.kernel)+  " C="+str(svc.C)+" gamma="+str(svc.gamma))
    scores = cross_val_score(svc, X_transformed, y_train, cv=5)
    print("交叉验证Accuracy： ", scores)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))