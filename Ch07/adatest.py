# *_*coding:utf-8 *_*
import adaboost
from numpy import *

# 单层决策树
datMat, classLabels = adaboost.loadSimpData()
D = mat (ones((5, 1)) / 5)
# print adaboost.buildStump(datMat, classLabels, D)

classifierArray = adaboost.adaBoostTrainDS(datMat, classLabels, 9)

print(classifierArray)

