import random
import numpy
from sklearn.metrics.pairwise import cosine_similarity

def randomSelect(rowProbDict):
    print(rowProbDict.values())
    print(rowProbDict.items())
    print(sum(rowProbDict.values() * 100))
#    upper = int(sum(rowProbDict.values() * 100))
    target = random.randint(0, int(sum(rowProbDict.values() * 100)))
    sum_ = 0
    for k, v in rowProbDict.items():
        sum_ += v*100
        if sum_ >= target:  # why??
            return k

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice



probDict={680: 0.75, 1194: 0.78, 334: 0.77, 1198: 0.78, 694: 0.73}
for i in range(20):
    print(randomSelect(probDict))

#print sorted(probDict.items(), key=lambda d: d[1])
s = sorted(probDict.items(), key=lambda d: d[1]) #renxl
print(s)


indexs = numpy.random.randint(0,10,5)
print(indexs)

#print cosine_similarity([1,2,3],[2,3,4]) #renxl
c = cosine_similarity([1,2,3],[2,3,4])
print(c)

# print(probDict)
# sort=sorted(probDict.items(), lambda x, y: cmp(x[1], y[1]),reverse=True)[:2]
# print(sort)
# for s in sort:
#     print(s)
#     probDict[s[0]]=s[1]
# print(probDict)