import regTrees
from numpy import *

trainMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))

myTree = regTrees.createTree(trainMat, ops = (1, 20))
yHat = regTrees.createForeCast(myTree, testMat[:, 0])
print (corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1])

myTree1 = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1, 20))
yHat = regTrees.createForeCast(myTree1, testMat[:, 0], regTrees.modelTreeEval)
print (corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1])


ws, X, Y = regTrees.linearSolve(trainMat)
print(ws)
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
print (corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1])