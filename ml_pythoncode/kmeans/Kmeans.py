#encoding:utf-8
#@Time : 2017/8/21 10:56
#@Author : JackNiu
from numpy import *


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension

        '''
        第j列最小值，rangeJ，rangeJ* random.rand(k,1)
        按列填充数据，列数据用最大值和最小值随机计算得出。
        这里的矩阵就是一个距离矩阵。也就是每一个点和该行代表的质心的距离。

        '''
        minJ = min(dataSet[:,j])
        maxJ=max(dataSet[:,j])


        rangeJ = float(maxJ-minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    '''
    这个clusterAssment  m*2： minIndex,minDist**2，保存的是每个点对应的质心，及其距离。
    '''
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                # 按照矩阵的优势 行进行计算
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        # print (centroids)

        '''
        根据上面计算的结果来更新质心。
        '''
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

def binKmeans(dataSet,k):
    '''
    二分k-means聚类算法
    SSE: 一种用于度量聚类效果的指标 误差平方和，SSE越小表示越接近于质心
    聚类效果也越好
    '''

    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    '''
    初始质心，均值
    '''
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList=[centroid0]
    for j in range(m):
        '''
        初始化: clusterAssment： [[0,dist],[0,dist],[0,dist]]
        '''
        clusterAssment[j,1]=distEclud(mat(centroid0),dataSet[j,:])**2
    while (len(centList)<k):
        lowerSSE=inf
        for i in range(len(centList)):
            '''
            x=nonzero(clusterAssment[:,0].A==i): 数组: 给定不为0的index
            x[0]: 返回这些index

            先根据质心0进行划分。
            '''
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            '''
            根据新的数据集得出的质心，和新的质心-距离矩阵,都是根据2-means做的。也就说是新的数据集的质心永远返回的是两个
            '''
            centroidMat, splitClustAss =kMeans(ptsInCurrCluster,2)
            '''
            SSE 的核心就是误差，是两个数据集的误差，总的=分开的和未分开的数据集误差之和。
            '''
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A !=i)[0],1])
            print ("sseSplit, and ssenotSplit: ",sseSplit,sseNotSplit,i)
            # 如果对该簇划分降低了SSE，则更新划分质心
            if (sseSplit+sseNotSplit)<lowerSSE:
                '''
                最好的分离点并不是现在的，而是上一个供划分子数据集对应的i
                质心: 2-means返回的质心。
                bestClustAss： 最开始的数据集划分成两个子数据集。
                '''
                bestCentToSplit=i
                bestNewCents=centroidMat
                bestClustAss = splitClustAss.copy()
                lowerSSE=sseSplit+sseNotSplit
        '''
        更新子数据集中对应的质心，2-means中0对应的是总质心，1对应的是新质心。len(centList)
        '''
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)

        print ("the bestCentToSpint is",bestCentToSplit,len(centList))
        print (" the len of bestClustAss is: ",len(bestClustAss))
        '''
        这才是更新质心，在质心List中，原来的[0]->[x,x]-> mean得出的，现在[0]->[x,x]-kmeans的0，并增加一个新的质心
        [1]->[x,x]->kmeans的1。
        '''
        centList[bestCentToSplit]=bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        # print (centList,len(centList),bestCentToSplit)
        '''
        并更新总数据集中的每一个数据点的质心，注意，每一次for运算都是以总数据为主。
        将kmeans返回的质心-->距离 替换原来的数据集中的质心-->距离，原来的都是[0,x]
        '''
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]=bestClustAss
        # print (len(clusterAssment))

    return centList,clusterAssment



#(1) K 均值聚类，并不能保证是全局最优，因此用二分K均值聚类提供一种量化方法SSE，误差平方和最小，则效果越好。
dataMat = mat(loadDataSet('testSet.txt'))
print(dataMat[0])
myCentorids,ClustAssing =binKmeans(dataMat,4)
# x=nonzero(ClustAssing[:,0]==0)
# print(x) # 以矩阵形式返回每一个值，A1将值转换为list。
# #[ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
# print(type(x),x[0])
print(ClustAssing)

