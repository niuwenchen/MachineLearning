#encoding:utf-8
#@Time : 2017/7/18 13:53
#@Author : JackNiu

from numpy import *

def loadDataSet(fileName):
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(list(fltLine))
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    '''
    数据集合，待切分的特征和该特征的某个值，通过数组过滤方式将上述数据集合切分得到
    :param dataSet:
    :param feature:
    :param value:
    :return:
    '''

    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):
    # 叶子节点, 一行数据的均值
    return mean(dataSet[:, -1])

def regErr(dataSet):
    # 均方差 * 总数目，行数也就是返回总方差
    return var(dataSet[:,-1])*shape(dataSet)[0]


def linearSolve(dataSet):
    '''
    将目标格式化成目标变量Y和自变量X
    X和Y用于执行简单的线性回归
    :param dataSet:
    :return:
    '''
    m,n = shape(dataSet)
    X= mat(ones((m,n)))
    Y= mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx) == 0.0:
        print("This matrix is sigular ,cannot do inverse \n")
        return
    #求ws 最简单的方法
    ws= xTx.I *(X.T *Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    # 预测值
    yHat = X*ws
    return sum(power(Y-yHat,2))




def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]; tolN=ops[1]  # tolN 数据子集的最小块
    # 如果这一列数据都是一样的，那就没办法根据列值进行划分，也就是说这一小块数据子集数据可以作为叶子节点。
    if len(set(dataSet[:,-1].T.tolist()[0])) ==1:
        print("1")
        return None,leafType(dataSet)
    m,n= shape(dataSet)
    S=errType(dataSet)
    bestS = inf; bestIndex=0;bestValue=0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):
            # 这是一个二分法，轮流迭代每一个值，直到找出最小的总方差
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS = errType(mat0)+errType(mat1)# 总方差是判断数据是否划分合适的最重要因素

            if newS < bestS:
                print("here",newS,tolS)
                bestIndex=featIndex
                bestValue = splitVal
                bestS = newS
    if (S- bestS) < tolS:
        print(S,bestS,tolS)
        # 尽管选出了最好的划分点，但是可能提升并不大，就直接作为叶子节点返回
        print('2')
        return None,leafType(dataSet)
    mat0,mat1= binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        print('3')
        return None,leafType(dataSet)
    return bestIndex,bestValue




def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    print(feat,val)
    if feat == None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    print(shape(lSet),shape(rSet))
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__=='dict')
def  getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    # 最后的 tree['right']  是一个值吗，最后的递归中，tree['right'] 和tree['left'都是值]
    return (tree['left']+ tree['right'])/2.0

def prune(tree,testData):
    # 如果测试集为空，则返回一个值,但是测试集为空的判断是0，有点不合理
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right'])) or isTree(tree['left']):
        print(tree['spInd'],tree['spVal'])
        # 测试集
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])

    if isTree(tree['left']): tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], lSet)

    # 到这里是最底层的节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2))+ \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree





dataMat = mat(loadDataSet('exp2.txt'))
x=createTree(dataMat,leafType=modelLeaf,errType=modelErr,ops=(1,10))
print(x)