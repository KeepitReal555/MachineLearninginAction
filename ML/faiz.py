# -*- coding: utf-8 -*-
#author: 王檬
import numpy as np
import random

def loadDataSet(filePath): #加载数据
    dataMat = []    # 特征矩阵
    labelMat = []   # 标签矩阵

    f = open(filePath)
    first_line = True

    # 分行读取数据
    for line in f.readlines():
        if first_line:
            first_line = False
        else:
            lineArr = line.strip('\n').split(',')
            tempArr = []
            i = 1 #从第2列开始
            while i < 29:
                tempArr.append(float(lineArr[i]))
                i += 1
            dataMat.append(tempArr)
            labelMat.append(int(lineArr[29]))

    print("*****************************************")
    print ("已经读取完" + filePath + "数据")
    print("*****************************************")
    f.close()
    return dataMat, labelMat

def  sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels): #梯度上升算法

    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()   #对标签列表转置，行变列
    m, n  = np.shape(dataMatrix) #矩阵大小
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))

    print("开始计算回归系数")

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=100): #改进的随机梯度上升算法
    print("*****************************************")
    print("开始使用改进的随机梯度上升算法")
    print("*****************************************")
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)  # initialize to all ones
    for j in range(numIter):
        print("开始第" + str(j+1) + "次循环......")
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights  #返回回归系数

def classifyVector(inX, weights):   #分类器
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5 : return 1.0
    else: return 0.0


def bankTest(): #测试错误率

    '''
    采用交叉验证方法测试错误率,在训练集中随机取出1000份数据验证算法准确性
    :return: 错误率
    '''

    #生成训练集
    trainingData = []; trainingLabels = []
    trainingData, trainingLabels = loadDataSet("train.csv")
    #生成回归系数
    trainWeights = stocGradAscent1(np.array(trainingData), trainingLabels, 4)

    m, n = np.shape(trainingData)

    #生成测试集
    testData = []; testLabels = []
    for i in range(1000):
        randIndex = int(random.uniform(0, m/2))
        print("index="+str(randIndex))
        testData.append(trainingData[randIndex])
        testLabels.append(int(trainingLabels[randIndex]))
        del (trainingData[randIndex])

    errorCount = 0; numTestVec = 0.0

    print("开始分类")
    x, y  = np.shape(testData)
    for num in range(x):
        numTestVec += 1.0
        print(num)
        if(int(classifyVector(np.array(testData[num]), trainWeights)) != int(testLabels[num])):
            print("分类错误...")
            errorCount += 1

    # 计算错误率
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is:" + str(errorRate))
    return errorRate

def banktestFinal():  #写result.csv
    '''
    最后的函数，直接运行，输出result.csv文件
    '''
    # 生成训练集
    trainingData = [];
    trainingLabels = []
    trainingData, trainingLabels = loadDataSet("train.csv") #加载训练集

    # 生成回归系数
    trainWeights = stocGradAscent1(np.array(trainingData), trainingLabels, 100) #循环100次后,AUC达到0.92
    #

    # 生成测试集
    # 由于与生成训练集有细微差别，所以没有直接用loadDataSet()函数
    testData = []; testLabels = []
    f = open("test.csv")
    first_line = True   #判断是否为第一行
    for line in f.readlines():
        if first_line:
            first_line = False
        else:
            lineArr = line.strip('\n').split(',')   # 将每行转为array
            tempArr = []
            i = 1  # 从第2列开始
            while i < 29:
                tempArr.append(float(lineArr[i]))   # 将特征量加入array
                i += 1
            testData.append(tempArr)    # 加入测试集
    m, n = np.shape(testData)

    #写result.csv
    fileName = "201531060251.csv"
    index = 199364  #测试级index是从这里开始的
    with open(fileName, 'w') as f:
        f.write('user_id,label\n')  #第一行
        for num in range(m):   #一行一行写
            if (int(classifyVector(np.array(testData[num]), trainWeights) == 1)):   #写标签为1的结果
                tempstring = str(int(index)+num) + ',' + str(1) + '\n'
            else:   #写标签为0的结果
                tempstring = str(int( index ) + num) + ',' + str(0) + '\n'
            f.write(tempstring)
    f.close() #关闭文件
