#encoding:utf-8
#@Time : 2017/8/29 10:50
#@Author : JackNiu

import numpy as np

def loadDataSet(fileName,delim="\t"):
    fp = open(fileName)
    stringArr = [line.strip().split(delim) for line in fp.readlines()]
    return np.mat(stringArr)

def pca(dataMat,topNfeat=9999999):
    meanVals = np.mean(dataMat,axis=0)
    meanremoved = dataMat-meanVals
    conMat =  np.cov(meanremoved,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(conMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
