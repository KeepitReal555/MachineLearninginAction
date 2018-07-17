from math import log
import operator

def calcShannonEnt(dataSet):
    '''
    计算给定数据集的香农熵
    :param dataSet:  数据集
    :return:
    '''
    numEntries = len(dataSet)   #计算数据集中实例的总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)   #以2为底求对数
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 待分配的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    '''
    retDataSet = []     #创建新的list对象，消除对list对象生命周期的影响
    for featVec in dataSet:
        #将符合特征的数据抽取出来
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet: 1.数据是由列表元素组成的列表
                    2.所有的列表元素都具有相同的数据长度
                    3.数据的最后一列是当前实例的类别标签
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)   #原始香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):    #遍历数据集中的所有特征
        featList = [example[i] for example in dataSet]  #数据集中第i个特征值和其所有可能存在的值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:    #遍历当前特征中的唯一属性值
            #计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #计算最好的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''

    :param classList:
    :return:出现次数最多的分类名称
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.Keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    '''
    创建树的函数代码
    :param dataSet:数据集
    :param labels:标签集
    :return:
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:    #遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}} #dict类型存储树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet] #得到列表包含的所有属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:    #遍历当前选择特征包含的所有属性值
        subLabels = labels[:]   #复制类标签，代替原始列表，不改变原列表内容
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat,
                                                               value),
                                                  subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

