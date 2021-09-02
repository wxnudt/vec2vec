#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy
import warnings
warnings.filterwarnings('ignore')
from gensim.models import word2vec
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import multiprocessing
from sklearn.covariance import EmpiricalCovariance
import faiss
from sklearn.preprocessing import normalize
from sklearn.preprocessing import maxabs_scale
from sklearn.preprocessing import minmax_scale
import datetime


# Using this method to save computation like our paper.
from vec2vec.node2vec_master import node2vec


def buildAdjacencyGraph(matrix, top_k):
    [rows, cols] = matrix.shape
    # print(rows, cols)

    # find the rows to compute the pairwise similarity
    dictColRowids={}
    for c in range(cols):
        listtmp=[]
        for r in range(rows):
            if(matrix[r][c]!=0 and r!=c):
                listtmp.append(r)
        dictColRowids[c]=listtmp


    # Computing the similarity between items in the matrix
    adjMatrix=numpy.zeros([rows,rows])

    # Build phash list for computing phash similarity
    # pHashList=[]
    # for r in  range(rows):
    #     pHashList.append(pHash(matrix[r]))

    for r in  range(rows):
        candidateList=[]
        for c in range(cols):
            if(matrix[r][c]!=0):
                candidateList.extend(dictColRowids[c])

        # candidateList = list(set(candidateList))
        candidateList = {}.fromkeys(candidateList).keys()
        #Compute the similarity with the candidate item. And get the top-k candidate items
        for cr in range(len(candidateList)):
            if(r!=cr):
                ## consine similarity
                adjMatrix[r][cr]=1.0-cosine(matrix[r],matrix[cr])

                # pHash Similarity  https://cloud.tencent.com/developer/article/1010909
                # adjMatrix[r][cr]=1 - hammingDist(pHashList[r],pHashList[cr])*1. / (32*32/4)

    # Find the top-k items for each row in the tmpmatrix
    # print(adjMatrix)
    # find the top-k items for each row in the adjmatrix
    [rows, cols] = adjMatrix.shape
    for row in range(rows):
        indexs = numpy.array(numpy.argsort(adjMatrix[row])[:rows - top_k])
        # print(indexs)
        for ind in indexs:
            adjMatrix[row][ind] = 0
    # print(adjMatrix)

    Graph = nx.convert_matrix.from_numpy_matrix(adjMatrix,parallel_edges=False,create_using=nx.DiGraph()).to_undirected()
    return Graph



def buildAdjacencyGraph2(matrix, top_k):
    # adjMatrix=cosine_similarity(matrix)
    # adjMatrix = euclidean_distances(matrix)
    # numpy.set_printoptions(suppress=True)
    # numpy.savetxt('D:/NSFC/project/data/input_matrix.txt',matrix,fmt='%1.2f')

    adjMatrix = cosine_similarity(matrix)
    # adjMatrix = euclidean_distances(matrix)
    [rows, cols] = adjMatrix.shape

    # Set the diagonal to be zero, there is no edge from a node to itself
    if(rows==cols):
        for r in range(rows):
            adjMatrix[r][r]=0

    # print(adjMatrix)
    #find the top-k items for each row in the adjmatrix
    for row in range(rows):
        indexs = numpy.array(numpy.argsort(adjMatrix[row])[:rows-top_k]) #renxl: 从小至大排列后的index
        # print(indexs)
        for ind in indexs:
            adjMatrix[row][ind] = 0 #renxl: top-k以外的元素置0

    # print(adjMatrix)

    # numpy.savetxt('D:/NSFC/project/data/input_adjMatrix.txt', adjMatrix, fmt='%1.2f')

    # graph=nx.convert_matrix.from_numpy_matrix(adjMatrix)
    graph = nx.convert_matrix.from_numpy_matrix(adjMatrix, parallel_edges=False, create_using=nx.DiGraph()).to_undirected()
    return graph



def buildAdjacencyGraph3(matrix, top_k):
    nn = NearestNeighbors(n_neighbors=top_k, metric='cosine', n_jobs=multiprocessing.cpu_count())
    nn.fit(matrix)
    adjMatrix = kneighbors_graph(nn, top_k, mode='distance', metric='cosine', n_jobs=multiprocessing.cpu_count()).toarray()
    [rows, cols] = adjMatrix.shape
    # Set the diagonal to be zero, there is no edge from a node to itself
    # if (rows == cols):
    #     for r in range(rows):
    #         adjMatrix[r][r] = 0
    # for row in range(rows):
    #     for ind in range(cols):
    #         if(adjMatrix[row][ind]!=0):
    #             adjMatrix[row][ind] = 1-adjMatrix[row][ind]
    numpy.where(adjMatrix>0,1-adjMatrix,0)
    graph = nx.convert_matrix.from_numpy_matrix(adjMatrix, parallel_edges=False, create_using=nx.DiGraph()).to_undirected()
    return graph



def buildCovarianceGraph(matrix, top_k):

    print("The shape of the input matrix: "+str(matrix.shape))
    adjMatrix = EmpiricalCovariance().fit(matrix).covariance_
    print("The shape of the ajdmatrix is: " +str(adjMatrix.shape))

    [rows, cols] = adjMatrix.shape

    # Set the diagonal to be zero, there is no edge from a node to itself
    if (rows == cols):
        for r in range(rows):
            adjMatrix[r][r] = 0

    # print(adjMatrix)
    # find the top-k items for each row in the adjmatrix
    for row in range(rows):
        indexs = numpy.array(numpy.argsort(adjMatrix[row])[:rows - top_k])
        # print(indexs)
        for ind in indexs:
            adjMatrix[row][ind] = 0

    # # find the smallest top-k items for each row in the adjmatrix
    # for row in range(rows):
    #     indexs = numpy.array(numpy.argsort(adjMatrix[row])[top_k])
    #     # print(indexs)
    #     for col in range(cols):
    #         if col not in indexs:
    #             adjMatrix[row][col] = 0


    graph = nx.convert_matrix.from_numpy_matrix(adjMatrix, parallel_edges=False, create_using=nx.DiGraph()).to_undirected()
    return graph



def buildNNGraphFromFAISS(matrix, top_k):
    # a = numpy.ascontiguousarray(( numpy.array([[0.0,1.0,1.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]).astype('float32')))
    # faiss.normalize_L2(a)
    # print(a)

    matrix = numpy.ascontiguousarray(matrix.astype('float32'))
    faiss.normalize_L2(matrix)
    print("The shape of the input matrix: "+str(matrix.shape))
    start = datetime.datetime.now()
    # index = faiss.IndexFlatIP(matrix.shape[1])
    quantizer = faiss.IndexFlatL2(matrix.shape[1])  # the other index
    index = faiss.IndexIVFFlat(quantizer,matrix.shape[1], 7, faiss.METRIC_INNER_PRODUCT)
    index.train(matrix)
    index.add(matrix)
    dists,inds = index.search(matrix, top_k)
    end = datetime.datetime.now()
    print('BuildNNGraphFromFAISS Finished in ' + str(end - start) + " s.")

    # print(dists)
    # print(inds)
    # distsNor = maxabs_scale(dists,axis=1)
    distsNor = minmax_scale(dists,axis=1)
    # print(distsNor)

    adjMatrix = numpy.zeros((matrix.shape[0], matrix.shape[0]))

    for row in range(distsNor.shape[0]):
        for col in range(distsNor.shape[1]):
            adjMatrix[row][inds[row][col]] = distsNor[row][col]

    for r in range(matrix.shape[0]):
        adjMatrix[r][r] = 0

    # adjMatrix = adjMatrix.astype(numpy.float32)
    print("The shape of the adjmatrix is: " +str(adjMatrix.shape))

    graph = nx.convert_matrix.from_numpy_matrix(adjMatrix, parallel_edges=False, create_using=nx.DiGraph()).to_undirected()
    return graph



def matrix2vec(matrix, dimensions, num_walks=10, walk_length=20, window_size=10, topk=10, p=1, q=1, num_iter=20):
    print("Matrix2vec p and q and topk: " + str(p) + " " + str(q)+" "+str(topk))
    # graph=buildAdjacencyGraph2(matrix,topk)
    graph = buildNNGraphFromFAISS(matrix, topk+1)
    G = node2vec.Graph(graph, False, p, q)

    start = datetime.datetime.now()
    G.preprocess_transition_probs()
    end = datetime.datetime.now()
    print('Preprocess_transition_probs Finished in ' + str(end - start) + " s.")

    walks = G.simulate_walks(num_walks, walk_length)
    # walks = [map(str, walk) for walk in walks]
    walks = [list(map(str, walk)) for walk in walks]
    end2 = datetime.datetime.now()
    print('Random Walk Finished in ' + str(end2 - end) + " s.")

    # walks=walks[0:1]
    print("Begin to train word2vec...")
    model = word2vec.Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0,
                              sg=1, workers=multiprocessing.cpu_count()-2, epochs=num_iter)

    # output='D:/NSFC/project/data/output_emb.m2v'
    # model.wv.save_word2vec_format(output)

    # print("Begin to copy datas from model...")
    embedding=numpy.ndarray((matrix.shape[0],dimensions))
    for index,word in enumerate(model.wv.index_to_key):
        # print (word+": ")
        # print(model.wv.vectors[index])
        embedding[int(word)]=numpy.array(model.wv.vectors[index])
    #embedding2 = [model.wv.vectors[index] for index, word in enumerate(model.wv.index_to_key)]
        # if(index%100==0):
        #     print("Copying data to embedding "+str(index))
    model = None
    return embedding