#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from siftdetector import detect_keypoints

def getMatchNum(matches,ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)


def read_libsvm_dataset(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]
	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)

# matrix=[[4,2,3,1,4,6],
#         [4,-2,3,8.0,9.1,3.7]]
# matrix=numpy.array([[1,2],[0.5,2.5],[2,4]])
# print(matrix)

# matrix=numpy.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[21,22,23,24],[25,26,27,28],[29,210,211,212]]])
# print(matrix)
# print(matrix.shape)
# matrix2=matrix.reshape([4,6])
# print(matrix2)
#
# # y_train,x_train=read_libsvm_dataset('D:/NSFC/project/data/20New/20news.train.bow')
# x_train,y_train=ds.load_svmlight_file('D:/NSFC/project/data/20New/test.txt')
#
# adjMatrix = cosine_similarity(x_train)
#
# print(x_train.shape)

# a=numpy.array(heapq.nsmallest(2, enumerate(matrix[0]), key=operator.itemgetter(1)))
# # # b=numpy.argsort(matrix[0])[::-1][:N] (N = len(matrix[0]))
# print(a)
# print(b)
# indexs = heapq.nsmallest(2, zip(matrix, itertools.count()))
# indexs = numpy.array(heapq.nsmallest(2, enumerate(matrix[0]), key=operator.itemgetter(1)))
# [rows, cols] = matrix.shape
# for row in range(rows):
#     indexs=numpy.array(numpy.argsort(matrix[row])[:2])
#     # print(indexs)
#     for ind in indexs:
#         matrix[row][ind]=0


# print(matrix)

# matrix2vec.buildAdjacencyGraph2(matrix,1)
# matrix2vec.matrix2vec(matrix,32,topk=1)



# CoilData = scipy.io.loadmat("D:/NSFC/project/data/coil20/COIL20.mat")  # Loading coil20.mat
# coil_x=CoilData['X']
# imageList=[]
# for item in coil_x:
#     imaM=numpy.zeros((128,128))
#     for i,it in enumerate(item):
#         imaM[int(i/128)][int(i%128)]=it;
#     imageList.append(imaM)


from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
print(type(X))
print(type(Y))
X.shape
Y.shape

logreg = linear_model.LogisticRegression(C=1e5)

a = logreg.fit(X, Y)
a.coef_  # 返回参数的系数
a.predict(X)  # 预测类别
a.predict_log_proba(X)  # 预测logit概率，即sigmoid函数值，发生率取对数
a.predict_proba(X)  # 预测概率P,时间发生的概率
a.score(X, Y)  # 预测精度





#创建SIFT特征提取器
# sift = cv2.xfeatures2d.SIFT_create()
# #创建FLANN匹配对象
# FLANN_INDEX_KDTREE=0
# indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
# searchParams=dict(checks=50)
# flann=cv2.FlannBasedMatcher(indexParams,searchParams)
#
# kp0, des0 = sift.detectAndCompute(imageList[0], None) #提取样本图片的特征
# for item in imageList:
#     kp2, des2 = sift.detectAndCompute(item, None)  # 提取比对图片的特征
#     matches = flann.knnMatch(des0, des2, k=2)  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
#     (matchNum, matchesMask) = getMatchNum(matches, 0.9)  # 通过比率条件，计算出匹配程度
#     matchRatio = matchNum * 100 / len(matches)
#     print("The match ratio of Item 0 and Item "+list.index(item)+" is "+str(matchRatio))

# HASH1=pHash(imageList[0])
# # HASH2=pHash('D:/NSFC/project/data/coil20/coil-20-proc/obj1__1.png')
# for index,image in enumerate(imageList):
#     HASH2 = pHash(image)
#     out_score = 1 - hammingDist(HASH1,HASH2)*1. / (32*32/4)
#
#     print(str(index)+" : "+str(out_score))


'''
adjMatrix=cosine_similarity(matrix)
[rows, cols] = adjMatrix.shape
print(adjMatrix.shape)

# Set the diagonal to be zero, there is no edge from a node to itself
if(rows==cols):
    for r in range(rows):
        adjMatrix[r][r]=0

print(adjMatrix)
 #find the top-k items for each row in the adjmatrix
indexs = numpy.array(heapq.nsmallest(1, enumerate(adjMatrix[0]), key=operator.itemgetter(1)))
print(indexs)
[indexrows, indexcols] = indexs.shape
for irow in range(indexrows):
    for icol in range(indexcols):
        adjMatrix[irow][indexs[irow][icol]]=0

'''