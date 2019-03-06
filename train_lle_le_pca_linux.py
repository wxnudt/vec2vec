#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing
import numpy as np
import argparse

from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA


def load_data(path):
    x_train, y_train = datasets.load_svmlight_file(path)
    x_train.todense()
    return x_train,y_train

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

    # data=load_data("D:/NSFC/project/data/movie/train.bow")
    data, label = load_data(args.input)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)


    models = []
    emb_size=128
    num_neighbors=64
    models.append(LocallyLinearEmbedding(n_neighbors=num_neighbors,n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
    models.append(SpectralEmbedding(n_neighbors=num_neighbors,n_components=emb_size,n_jobs=multiprocessing.cpu_count()))
    models.append(PCA(n_components=emb_size))

    model_names=['lle','le','pca'] # names corresponding to models

    for index, embedding in enumerate(models):
        X_transformed = embedding.fit_transform(data.toarray())
        # print(X_transformed)
        if(len(X_transformed)==len(label)):
            #Combine the array X_transformed and label
            x_data_label=[]
            for in2,item in enumerate(X_transformed):
                itemTemp=[]
                for in3, val in enumerate(item):
                    pos=in3+1
                    itemTemp.append(str(pos)+":"+'{:01.6f}'.format(val))

                x_data_label.append(np.append('{:g}'.format(label[in2]),itemTemp))
                #print(np.append('{:g}'.format(label[in2]),itemTemp))

        outputpath=args.output+'.'+model_names[index]+'.emb'+str(emb_size)+'.neigh'+str(num_neighbors)+'.sort.tmp'
        np.savetxt(outputpath, x_data_label, fmt="%s")

