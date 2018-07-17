import matplotlib.pyplot as plt
import regression
from numpy import *
xArr, yArr = regression.loadDataSet('ex0.txt')
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)

xMat = mat(xArr)
yMat = mat(yArr)
strInd = xMat[:,1].argsort(0)
xSort = xMat[strInd][:,0,:]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1], yHat[strInd])
ax.scatter(xMat[:,1].flatten().A[0],
           yMat.T[:,0].flatten().A[0],
           s = 2, c = 'red')
plt.show()