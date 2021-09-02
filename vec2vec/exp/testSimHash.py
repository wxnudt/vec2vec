from simhash import Simhash,SimhashIndex
import numpy

# Using this method to save computation like our paper.
def get_features(array):
    features=[]
    for item in array:
        if(item==1.00):
            item=0.999
        if (item == 0.0):
            item = 0.000
        features.append(str(round(item, 1)).partition('.')[2])
    return features

# For 32-bit fingerprints
def buildSimHashIndex(matrix):

    hashDictArray=[]
    for num in range(4):
        hashDictArray.append({})

    for index,item in enumerate(matrix):
        hash=Simhash(get_features(item),f=32).value
        hashstr=bin(hash)

        for num in range(4): # For 32-bit fingerprints
            key=int(hashstr[num*8+2:(num+1)*8+2])

            if(hashDictArray[num].has_key(key)):
                hashDictArray[num][key].append(index)
            else:
                hashDictArray[num][key]=[]
                hashDictArray[num][key].append(index)

    return hashDictArray



def findNeighborVectors(simHashIndex, matrix, vector, topk=10):
    hash = Simhash(get_features(vector), f=32)
    hashstr = bin(hash.value)

    nearVectorArray=[]

    for num in range(4):  # For 32-bit fingerprints
        key = hashstr[num * 8 + 2:(num + 1) * 8 + 2]
        # if (simHashIndex[num].has_key(key)):
        #     nearVectorArray.extend(simHashIndex[num][key])
        nearVectorArray.extend(simHashIndex[num].get(int(key)))

    sortedDict= {}
    for index in nearVectorArray:
        index=int(index)
        vec=matrix[index]
        vechash=Simhash(get_features(vector), f=32)
        sortedDict[index]=hash.distance(vechash)

    sorted(sortedDict.items(), lambda x, y: cmp(x[1], y[1]))
    sortindexs=sortedDict.keys()[:topk]

    return sortindexs





def testSimHash(matrix):
    for index,item in enumerate(matrix):
        print index,item
        print(get_features(item))
        print(Simhash(get_features(item)).value)
        print(bin(Simhash(get_features(item)).value))






data=[[0.533,0.133,0.2555,0,0.88,0.78,0,0,0,0],[0.4811,0.4122,0.1133,0.213,0.11,0.22,0,0,0,0],[0.524,0.413,0.113,0.213,1.212,2.210,0,0,0,0,],[0.933,0.133,0.2555,0,0.88,0.78,0,0,0,0],[0.91555,0.1266,0.266,0.0166,0.89,0.79,0,0,0,0],[0.633,0.233,0.2555,0,0.88,0.78,0,0,0,0],[0.133,0.333,0.4555,0,0.88,0.78,0,0,0,0]]
# testSimHash(data)
sh1=Simhash(get_features(data[0]),f=32)
sh2=Simhash(get_features(data[1]))
sh3=Simhash(get_features(data[2]))
sh4=Simhash(get_features(data[3]))

simIndex=buildSimHashIndex(data)
print(simIndex)
query=[0.533,0.133,0.2555,0,0.88,0.78,0,0,0,0]
index=findNeighborVectors(simIndex,data,query,3)
print(index)


# sh1=Simhash(data[0])
# sh2=Simhash(data[1])
# sh3=Simhash(data[2])
# sh4=Simhash(data[3])
# print(sh1.distance(sh2))
# print(sh1.distance(sh3))
# print(sh1.distance(sh4))
# print(sh2.distance(sh3))
# print(sh2.distance(sh4))
# print(sh3.distance(sh4))






