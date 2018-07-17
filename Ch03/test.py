from . import trees

#计算给定数据集的香农熵
myDat, labels = trees.createDataSet()
trees.alcShannonEnt(myDat)

'''
>>> import trees
>>> myDat, labels = trees.createDataSet()
>>> myTree = trees.createTree(myDat, labels)
>>> myTree
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
>>> 

'''

'''
>>> reload(treePlotter)
<module 'treePlotter' from '/Users/Faiz/PycharmProjects/MachineLearningIASelf/Ch03/treePlotter.py'>
>>> treePlotter.createPlot()

'''