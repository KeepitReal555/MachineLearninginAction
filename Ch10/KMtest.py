import kMeans
from numpy import *

datMat = mat(kMeans.loadDataSet('testSet.txt'))
# print kMeans.randCent(datMat, 2)
# print kMeans.distEclud(datMat[0], datMat[1])

myCentroids, clustAssing = kMeans.kMeans(datMat, 4)
# print myCentroids
# print clustAssing