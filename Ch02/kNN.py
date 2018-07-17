from numpy import *
import operator
import os


def createDataSet():
    '''
    创建数据集和标签
    :return:
    '''
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def gaussian(dist, sigma = 10.0):
    weight = exp(-dist**2/(2*sigma**2))
    return weight

def classify0(inX, dataSet, labels, k):
    '''
    分类器实例
    :param inX: 需分类的输入向量
    :param dataSet: 训练样本集
    :param labels:  标签向量
    :param k:   最近邻居数目
    :return: 频率最高的元素标签
    '''
    dataSetSize = dataSet.shape[0]  #获取数组行数

    #计算欧几里得距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #xA0-xB0,将inX变为等同于dataSet的矩阵
    sqDiffMat = diffMat**2  #(xA0-xB0)^
    sqDistances = sqDiffMat.sum(axis=1) #(xA0-xB0)^+(xA1-xB1)^,对2维求和，变为1维数组
    distances = sqDistances**0.5    #√((xA0-xB0)^+(xA1-xB1)^)

    sortedDistIndicies = distances.argsort()    #增序排列

    # classCount = {}     #集合
    # #选取距离最小的k个点的标签
    # for i in range(k):
    #     voteIlabel = labels[sortedDistIndicies[i]]
    #     classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #https://blog.csdn.net/weixin_38705903/article/details/79231551
    #
    # #排序标签
    # sortedClassCount = sorted(classCount.items(),
    #                           key=operator.itemgetter(1),
    #                           reverse=True) #降序排列
    #
    # return sortedClassCount[0][0]

    # 算法优化
    # 高斯衰减优化
    # 权重
    weightCount = {}
    for i in range(k):
        weight = gaussian(distances[sortedDistIndicies[i]])
        weightCount[labels[sortedDistIndicies[i]]] = weightCount.get(labels[sortedDistIndicies[i]], 0) + weight
    sortedWeightCount = sorted(weightCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedWeightCount[0][0]




def file2matrix(filename):
    '''
    将文本记录转为Numpy的解析程序
    :param filename: 文件名字符串
    :return: 训练样本集，类标签向量
    '''
    fr = open(filename)
    arraryOLines = fr.readlines()   #得到文件行数
    numberOLines = len(arraryOLines)
    returnMat = zeros((numberOLines,3)) #创建返回的Numpy矩阵
    classLabelVector = []
    index = 0
    for line in arraryOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]  #赋值index行的所有元素 https://blog.csdn.net/Strive_0902/article/details/78225691
        classLabelVector.append(int(listFromLine[-1])) #赋值类标签
        index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''
    归一化特征值
    :param dataSet:
    :return: 0到1区间的值,最大最小值差值，最小值
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    '''
    测试算法
    :return:
    '''
    hoRation = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRation)

    # errorCount = 0.0
    # for i in range(numTestVecs):
    #     classifierResult = classify0(normMat[i, :],
    #                                  normMat[numTestVecs:m, :],
    #                                  datingLabels[numTestVecs:m],3)
    #     print("the classifier came back with %d, the real answer is: %d"
    #           % (classifierResult, datingLabels[i]))
    #     if (classifierResult != datingLabels[i]): errorCount += 1.0
    # print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

    #算法优化
    #交叉验证：取一份测试，其他为训练集
    k = 2   #kNN算法中k的取值
    while k<21:
        sum_errorRate = 0.0
        avg_errorRate = 0.0
        for i in range(10): #交叉验证
            errorCount = 0.0
            start = int(numTestVecs*i)
            end = int(start+numTestVecs)
            j = start
            while j < end:
                classifierResult = classify0(normMat[j, :],
                                             concatenate((normMat[0:start, :], normMat[end:m, :]), axis=0),
                                             concatenate((datingLabels[0:start], datingLabels[end:m]), axis=0),
                                             k)
                if (classifierResult != datingLabels[j]): errorCount += 1.0
                j += 1
            #每个j的错误率累加
            sum_errorRate += (errorCount / float(numTestVecs))

        #在每个i下的算平均错误率
        avg_errorRate = sum_errorRate / 10.0
        #输出
        print("when k = %d error rate is: %f" % (k, avg_errorRate))
        print("---------------------------------------")
        k += 1

def classifyPerson():
    '''
    约会网站预测函数
    :return:
    '''
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("frequent filer miles earned per year?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print ("You will probably like this person: ", resultList[classifierResult - 1])

def img2vector(filename):
    '''
    将图像转换为测试向量
    :param filename:
    :return:
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    '''
    手写数字识别系统的测试代码
    :return:
    '''
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits') #获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    #从文件名解析分类数字
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))




