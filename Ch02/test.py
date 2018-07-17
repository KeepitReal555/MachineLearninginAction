
'''
>>>import kNN

>>>datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

>>>import matplotlib
>>>import matplotlib.pyplot as plt
>>>fig = plt.figure()
>>>ax = fig.add_subplot(111)
>>>ax.scatter(datingDataMat[:,1], datingDataMat[:, 2]) #散点图使用矩阵的第二，三列数据
>>>plt.show()

>>>import numpy
>>>ax.scatter(datingDataMat[:,1], datingDataMat[:, 2], 15.0*numpy.array(datingLabels), 15.0*numpy.array(datingLabels)) #个性化标记散点
>>>plt.show()

#归一化特征值
>>>from imp import reload
>>>reload(kNN)
>>>normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
>>>normMat
>>>ranges
>>>minVals

#测试算法
>>>reload(kNN)
>>>kNN.datingClassTest()

#约会网站预测函数
>>>reload(kNN)
>>>kNN.classifyPerson()

#将图像转换为测试向量
>>>testVector = kNN.img2vector('digits/testDigits/0_13.txt')
>>>testVector[0, 0:31]
>>>testVector[0, 32:63]

#手写数字识别系统的测试代码
>>>kNN.handwritingClassTest()

'''
