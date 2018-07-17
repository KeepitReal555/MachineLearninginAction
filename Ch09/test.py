import regTrees
from numpy import *
import matplotlib.pyplot as plt

myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
print(regTrees.createTree(myMat))
plt.plot(myMat[:,0],myMat[:,1], 'ro')
plt.show()

myDat1 = regTrees.loadDataSet('ex0.txt')
myMat1 = mat(myDat1)
print(regTrees.createTree(myMat1))
plt.plot(myMat1[:,1],myMat1[:,2], 'ro')
plt.show()
