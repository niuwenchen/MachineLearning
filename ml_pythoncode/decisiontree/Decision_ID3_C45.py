#encoding:utf-8
#@Time : 2017/5/27 16:11
#@Author : JackNiu

from numpy import *
import math
import copy
import pickle as pickle

class ID3Tree(object):
    def __init__(self):
        self.tree={}
        self.dataSet={}
        self.labels=[]
    def  loadDateSet(self,path,labels):
        recordlist=[]
        fp=open(path,'r')
        content= fp.read()
        fp.close()
        rowlist = content.splitlines()
        recordlist=[row.split('\t') for row in rowlist if row.strip()]
        self.dataSet=recordlist
        self.labels=labels
    def train(self):
        labels= copy.deepcopy(self.labels)
        self.tree = self.buildTree(self.dataSet,labels)

    def buildTree(self,dataSet,labels):
        cateList = [data[-1] for data in dataSet]
        # 程序终止条件1:如果classList只有一种决策标签，停止划分，返回这个决策标签
        if cateList.count(cateList[0]) == len(cateList):
            return cateList[0]
        # 程序终止条件2： 如果数据集的第一个决策标签只有一个，则返回这个决策标签
        if (len(dataSet[0])==1):
            return self.maxCate(cateList)

        # 算法核心
        bestFeat = self.getBestFeat(dataSet)
        bestFeatLabel = labels[bestFeat]
        tree={bestFeatLabel:{}}
        del(labels[bestFeat])

        # 抽取最优特征轴的列向量
        uniqueVals= set([data[bestFeat] for data in dataSet])
        for value in uniqueVals:
            subLabels = labels[:]    #将删除后的特征类别建立子类别集
            splitDataset = self.splitDataSet(dataSet,bestFeat,value)
            subTree = self.buildTree(splitDataset,subLabels)
            tree[bestFeatLabel][value]=subTree
        return tree




    def getBestFeat(self,dataSet):
        numFeatures=len(dataSet[0])-1
        baseEntropy = self.computeEntropy(dataSet)  # 基础熵：元数据的熵
        bestInfoGain =0.0                           # 初始化最优的信息增益
        bestFeature=-1
        # 外循环，遍历数据集各列，获取最优特征轴
        for i in range(numFeatures):
            uniqueVals=set([data[i] for data in dataSet])  # 抽取第i列的列向量
            newEntropy= 0.0
            for value in uniqueVals:
                subDataSet =self.splitDataSet(dataSet,i,value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * self.computeEntropy(subDataSet)
            infoGain =baseEntropy-newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain= infoGain
                bestFeature = i
        return bestFeature



    def computeEntropy(self,dataSet):
        datalen= float(len(dataSet))
        cateList = [data[-1] for data in dataSet]
        items = dict([(i,cateList.count(i)) for i in cateList])
        infoEntropy=0.0
        for key in items:
            print("数据集中的类别 %s"%key)
            prob = float(items[key])/datalen
            infoEntropy -= prob*math.log(prob,2)
        return infoEntropy

    # 划分数据集：分隔数据集： 删除特征轴所在的数据列，返回剩余的数据集
    def splitDataSet(self,dataSet,axis,value):
        retList =[]
        for featVec in dataSet:
            if featVec[axis] == value:
                retvec=featVec[:axis]  #list操作，提取 0~(axis-1)的元素
                retvec.extend(featVec[axis+1:])   # 将特征列之后的数据加上
                retList.append(retvec)
        return retList

    def storeTree(self,inputTree,filename):
        fw=open(filename,'w')
        pickle.dump(inputTree,fw)
        fw.close()

    def gradTree(self,filename):
        fr=open(filename)
        return pickle.load(fr)

    def predict(self,inputTree,featLabels,testVec):
        root= inputTree.keys()[0]
        secondDict=inputTree[root]
        featIndex = featLabels.index(root)
        key=testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat,dict):
            classLabel = self.predict(valueOfFeat,featLabels,testVec)
        else:
            classLabel= valueOfFeat
        return classLabel


    def getC45BestFeat(self,dataSet):
        num_feats=len(dataSet[0][:-1])
        totality = len(dataSet)
        BaseEntropy = self.computeEntropy(dataSet)
        ConditionEntropy=[]
        SpiltInfo=[]
        allFeatVList=[]

        for f in range(num_feats):
            featList = [example[f] for example in dataSet]
            [splitI,featureValueList] =self.computeSpiltInfo(featList)
            allFeatVList.append(featureValueList)
            SpiltInfo.append(splitI)
            resultGain =0.0
            for value in featureValueList:
                subSet = self.splitDataSet(dataSet,f,value)
                appearNum = float(len(subSet))
                subEntroy = self.computeEntropy(subSet)
                resultGain  += (appearNum/totality)* subEntroy
            ConditionEntropy.append(resultGain)

        infoGainArray= BaseEntropy*ones(num_feats)-array(ConditionEntropy)
        infoGainRatio = infoGainArray/array(SpiltInfo)
        bestFeatureindex = argsort(-infoGainRatio)[0]
        return bestFeatureindex,allFeatVList[bestFeatureindex]

    def computeSpiltInfo(self,featureVList):
        numEntroies = len(featureVList)
        featureValueSetList = list(set(featureVList))
        valueCounts= [featureVList.count(featVec) for featVec in featureValueSetList]
        pList = [float(item)/numEntroies for item in valueCounts]
        lList =[item*math.log(item,2) for item in pList]
        spiltInfo = -sum(lList)
        return spiltInfo, featureValueSetList




dtree = ID3Tree()
dtree.loadDateSet("dataset.dat",["age","revenue","student","credit"])
dtree.train()
print(dtree.tree)
















