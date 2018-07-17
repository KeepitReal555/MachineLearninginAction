import matplotlib.pyplot as plt
import regression
from numpy import *
xArr, yArr = regression.loadDataSet('ex0.txt')
ws = regression.standRegres(xArr, yArr)
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],
           yMat.T[:,0].flatten().A[0],)
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:,1], yHat)
plt.show()
