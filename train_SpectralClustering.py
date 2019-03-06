#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing
import numpy as np
import argparse

from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn import metrics


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

    inpath="D:/NSFC/project/data/movie/movie.train.bow.pca.emb512.neigh64.sort.tmp"
    data, label=load_data(inpath)
    # data, label = load_data(args.input)
    # print(label)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # outputpath=args.input+".SpeClu"
    outputpath=inpath+".SpeClu"
    out=[]

    for gamma in (1e-6,0.00001, 0.0001,0.001,0.01, 0.1, 1):
        y_pred = SpectralClustering(n_clusters=2, gamma=gamma,n_jobs=multiprocessing.cpu_count()-1).fit_predict(data)
        print("Adjusted Rand Index with gamma="+str(gamma)+" score:"+str(metrics.adjusted_rand_score(label,  y_pred)))
        out.append("Adjusted Rand Index with gamma="+str(gamma)+" score:"+str(metrics.adjusted_rand_score(label,  y_pred)))
    np.savetxt(outputpath, out, fmt="%s")





