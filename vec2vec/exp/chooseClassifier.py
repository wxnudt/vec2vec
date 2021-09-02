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
import cPickle as pickle
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
    train, val, test = pickle.load(f)
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
    # x_train, y_train, x_test, y_test = read_data("D:/NSFC/project/data/MNIST/origData/mnistpklgz/mnist.pkl.gz")

    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/20NewLong/train.bow')
    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/movie/train.bow')
    x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/GoogleSnippets/traintest.txt.bow')
    # x_train2, y_train2 = ds.load_svmlight_file('D:/NSFC/project/data/stackoverflow/title_label.txt.bow')

    # x_train, x_te, y_train, y_te = train_test_split(x_train2, y_train2, train_size=500)


    # x_train, y_train = unpickle("D:/NSFC/project/data/cifar10/cifar-10-batches-py/data_batch_1")

    print("Load dataset finished...")

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # x_train=x_train[0:5000, :]
    # y_train = y_train[0:5000]

    x_train = x_train2.toarray()
    y_train = y_train2
    # x_train=MinMaxScaler().fit_transform(x_train)
    # x_train=x_train[0:2000, :]
    # y_train = y_train[0:2000]

    print(x_train.shape)


    #Using KNN classifier to test the result with cross_validation
    # x_tr, x_te, y_tr, y_te = train_test_split(X_transformed, y_train, test_size=0.1)
    time_knn_begin=datetime.datetime.now()
    knn = KNeighborsClassifier()
    param = {"n_neighbors": [1, 3, 5, 7, 11]}  # 构造一些参数的值进行搜索 (字典类型，可以有多个参数)
    gc = GridSearchCV(knn, param_grid=param, cv=4)
    gc.fit(x_train, y_train)
    knn = gc.best_estimator_
    print("The best parameter: n_neighbors=" + str(knn.n_neighbors))
    scores = cross_val_score(knn, x_train, y_train, cv=4)
    time_knn_end = datetime.datetime.now()
    print("Classifier KNN Finished in " + str(time_knn_end - time_knn_begin) + " s.")
    print("交叉验证Accuracy： ", scores)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    #print("在测试集上准确率：", gc.score(x_te, y_te))  # 0.432624113475
    #print("在交叉验证当中最好的结果：", gc.best_score_)  # 0.390684110971 并没有使用测试集。(验证集是从训练集中分的)
    #print("选择最好的模型是：", gc.best_estimator_)  # KNeighborsClassifier(algorithm="auto", n_neighbors=10)
    #print("每个超参数每次交叉验证的结果：", gc.cv_results_)
    #knn.predict(x_te)
    #print("最优参数的结果是： ",knn.score(x_te,y_te))

    # scores = cross_val_score(knn, X_transformed, y_train, cv=4)
    # print("交叉验证Accuracy： ",scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print("train_accuracy :" + str(knn.score(X_transformed, y_train)))

    # print("Begin to search for better parameters...")
    time_svc_begin = datetime.datetime.now()
    clf = svm.SVC()
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4,1e-3, 0.01,0.1,1],
    #         #                              'C': [1]},
    #         #                              {'kernel': ['linear'], 'C': [1],'gamma': [1e-4,1e-3, 0.01,0.1,1]}]
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [1, 2, 8, 32, 128, 512]}]
    # tuned_parameters = [
    #     {'kernel': ['linear'], 'C': [2, 8, 32, 128, 512, 2048, 8192],
    #      'gamma': [0.0078125, 0.03125, 0.125, 0.5]}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=4, n_jobs=multiprocessing.cpu_count()-1)
    # clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=4, n_jobs=multiprocessing.cpu_count() - 1, verbose=5)
    clf.fit(x_train, y_train)
    svc=clf.best_estimator_

    print("The best parameter is：kernel="+str(svc.kernel)+" C="+str(svc.C)+" gamma="+str(svc._gamma))

    scores2 = cross_val_score(svc, x_train, y_train, cv=4)
    time_svc_end = datetime.datetime.now()
    print("Classifier SVC Finished in " + str(time_svc_end - time_svc_begin) + " s.")
    print("LinearSVC Cross validation Accuracy： ", scores2)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores2.mean(), scores2.std() * 2))

    # para_c_list= [2, 8, 32, 128, 512, 32768]
    # # para_gamma_list=[0.0078125, 0.03125, 0.125, 0.5]
    # for para_c in para_c_list:
    #     svc = svm.SVC(C=para_c,kernel='linear')
    #
    #     grid_search_time = datetime.datetime.now()
    #     print("The best parameter is：kernel=" + str(svc.kernel) + " C=" + str(svc.C) + " gamma=" + str(svc.gamma) +" time="+str(grid_search_time))
    #
    #     scores = cross_val_score(svc, X_transformed, y_train, cv=4)
    #     print("交叉验证Accuracy： ", scores)
    #     print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
