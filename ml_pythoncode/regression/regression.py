#encoding:utf-8
#@Time : 2017/7/17 23:14
#@Author : JackNiu

from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 计算最佳拟合直线
# ws=(X^(T)X)^(-1) X^(T) y
# 是根据求平方误差,对w求导得出的
def standReges(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat  #行列式
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

# def testws():
#     x,y = loadDataSet('ex0.txt')
#     ws= standReges(x,y)
#     print(ws)



# lwlr 也是没无监督学习
def lwlr(testPoint,xArr,yArr,k=1.0):
    # testPoint 的意义：待定的预测矩阵的某一行
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    # 只是对角线元素，假定为方阵
    weights = mat(eye((m)))
    for j in range(m):
        diffMat= testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))

    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print("This matrix is sigular ,cannot do inverse")
        return
    ws = xTx.I * ( xMat.T * (weights * yMat))
    return testPoint * ws

# lwlr训练，需要随着k的大小进行训练
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i]= lwlr(testArr[i],xArr,yArr,k)
    return yHat


def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()



# 岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print('This matrix is sigular ,cannot do inverse')
        return
    ws= denom.I *(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat =mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat- yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat =zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]= ws.T
    return wMat



x,y= loadDataSet('abalone.txt')
weight=ridgeTest(x,y)
print(weight)