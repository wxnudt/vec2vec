#!/usr/bin/env python
# -*- coding: utf-8 -*-

from node2vec_master import node2vec
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy
from gensim.models import word2vec


# To compute effectively for large-scale matrices based on the inverted index
def buildAdjacencyGraph(matrix, top_k):
    [rows, cols] = matrix.shape

    # find the rows for computing the pairwise similarity
    dictColRowids={}
    for c in range(cols):
        listtmp=[]
        for r in range(rows):
            if(matrix[r][c]!=0 and r!=c):
                listtmp.append(r)
        dictColRowids[c]=listtmp

    # Computing the similarity between items in the matrix
    adjMatrix=numpy.zeros([rows,rows])
    for r in  range(rows):
        candidateList=[]
        for c in range(cols):
            if(matrix[r][c]!=0):
                candidateList.extend(dictColRowids[c])

        # candidateList = list(set(candidateList))
        candidateList = {}.fromkeys(candidateList).keys()
        #Compute the similarity with the candidate items
        for cr in range(len(candidateList)):
            if(r!=cr):
                adjMatrix[r][cr]=1.0-cosine(matrix[r],matrix[cr])

    # find the top-k items for each row in the adjmatrix
    [rows, cols] = adjMatrix.shape
    for row in range(rows):
        indexs = numpy.array(numpy.argsort(adjMatrix[row])[:rows - top_k])
        for ind in indexs:
            adjMatrix[row][ind] = 0

    Graph = nx.convert_matrix.from_numpy_matrix(adjMatrix,parallel_edges=False,create_using=nx.DiGraph()).to_undirected()
    return Graph


def buildAdjacencyGraph2(matrix, top_k):

    adjMatrix = cosine_similarity(matrix)
    [rows, cols] = adjMatrix.shape

    # Set the diagonal to be zero, there is no edge from a node to itself
    if(rows==cols):
        for r in range(rows):
            adjMatrix[r][r]=0

    for row in range(rows):
        indexs = numpy.array(numpy.argsort(adjMatrix[row])[:rows-top_k])
        for ind in indexs:
            adjMatrix[row][ind] = 0

    graph = nx.convert_matrix.from_numpy_matrix(adjMatrix, parallel_edges=False, create_using=nx.DiGraph()).to_undirected()
    return graph



def matrix2vec(matrix, dimensions, num_walks=10, walk_length=20, window_size=10, topk=10, p=1, q=1, num_iter=20):
    graph=buildAdjacencyGraph2(matrix,topk)
    G = node2vec.Graph(graph, False, p, q)

    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    walks = [list(map(str, walk)) for walk in walks]

    model = word2vec.Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=8, iter=num_iter)
    # output='D:/NSFC/project/data/output_emb.m2v'
    # model.wv.save_word2vec_format(output)

    embedding=numpy.ndarray((matrix.shape[0],dimensions))
    for index,word in enumerate(model.wv.index2word):
        embedding[int(word)]=numpy.array(model.wv.vectors[index])

    model = None
    return embedding
