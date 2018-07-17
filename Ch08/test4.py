import matplotlib.pyplot as plt
import regression
from numpy import *

xArr, yArr = regression.loadDataSet('abalone.txt')
# a = regression.stageWise(xArr, yArr, 0.01, 200)
# b = regression.stageWise(xArr, yArr, 0.01, 50000)

xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regression.regularize(xMat)
yM = mean(yMat, 0)
yMat = yMat - yM
weights = regression.standRegres(xMat, yMat.T)
# weights.T
testWeights = regression.stageWise(xArr, yArr, 0.005, 1000)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(testWeights)
plt.show()