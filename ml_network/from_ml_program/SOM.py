#encoding:utf-8
#@Time : 2017/6/19 17:54
#@Author : JackNiu

from numpy import *
import  matplotlib.pyplot as plt


class SOM(object):
    def __init__(self):
        self.lratemax=0.8
        self.lratemin = 0.05
        self.rmax=5.0
        self.rmin = 0.5
        self.steps = 1000
        self.lratelist=[]
        self.rlist=[]
        self.w=[]
        self.M=2
        self.N=2
        self.dataMat=[]
        self.classLabel=[]

    def normalize(self,dataMat):
        [m, n] = shape(dataMat)
        for i in range(n - 1):
            dataMat[:, i] = (dataMat[:, i] - mean(dataMat[:, i])) / (std(dataMat[:, i]) + 1.0e-10)  # (x-xba)/标准差
        return dataMat

    # 计算矩阵各向量之间的距离--欧氏距离
    # 两个矩阵的欧式距离计算过程： 行列距离X 行  与 Y列
    def distEclud(self,matA,matB):
        ma, na = shape(matA)
        mb, nb = shape(matB)
        rtnmat = zeros((ma, nb))
        for i in range(ma):
            for j in range(nb):
                # 范数，默认是二范数，就是欧氏距离。
                rtnmat[i, j] = linalg.norm(matA[i, :] - matB[:, j].T)
        return rtnmat

    def loadDataSet(self,fileName):
        numFeat = len(open(fileName).readline().split('\t')) - 1
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            lineArr.append(float(curLine[0]))
            lineArr.append(float(curLine[1]))
            self.dataMat.append(lineArr)
        self.dataMat = mat(self.dataMat)

    def  init_grid(self):
        k=0  # 构建第二层网络模型
        grid = mat(zeros((self.M*self.N,2)))
        for i in range(self.M):
            for j in range(self.N):
                grid[k,:]=[i,j]
                k +=1
        # 这里在用的时候应该需要一个转置
        return grid

    # 学习率和学习半径函数
    def ratecalc(self, indx):
        lrate = self.lratemax - (float(indx) + 1.0) / float(self.steps) * (self.lratemax - self.lratemin)
        r = self.rmax - (float(indx) + 1.0) / float(self.steps) * (self.rmax - self.rmin)
        return lrate, r


    def  train(self):
        dm,dn =  shape(self.dataMat)
        normDataset= self.normalize(self.dataMat)
        grid = self.init_grid()
        self.w = random.rand(dn,self.M*self.N)  # 随机初始化两层之间的向量权重，2*4维
        distM = self.distEclud
        # 迭代求解
        if self.steps < 5*dm:
            self.steps= 5*dm
        for i in range(self.steps):
            lrate, r=  self.ratecalc(i)  # 计算当前迭代次数下的学习率和分类半径
            #2） 随机生成样本索引，并抽取一个样本
            k = random.randint(0,dm)
            print(k)
            mysample = normDataset[k,:]
            #3) 计算最优节点，返回最小距离的索引值
            # 这个距离 是求的是向量权重和样本的最小距离，很奇诡
            minIndx = (distM(mysample,self.w)).argmin()
            print('minIndx',minIndx)
            #4) 计算邻域
            d1= ceil(minIndx/self.M)
            d2= mod(minIndx,self.M)
            print('d1d2',d1,d2)

            distMat = distM(mat([d1,d2]),grid.T)
            print(distMat<r)
            #5)　获取邻域内的所有节点
            nodelindx = nonzero(distMat<r)[1]
            for j in range(shape(self.w)[1]):
                if sum(nodelindx == j):
                    self.w[:,j] =self.w[:,j] + lrate*(mysample[0]-self.w[:,j])

        self.classLabel = list(range(dm))
        for i in range(dm):
            self.classLabel[i] = distM(normDataset[i, :], self.w).argmin()
        self.classLabel = mat(self.classLabel)


    def showCluster(self,plt):
        lst = unique(self.classLabel.tolist()[0])
        i=0
        for cindx in lst:
            myclass =nonzero(self.classLabel==cindx)[1]
            xx =self.dataMat[myclass].copy()
            if i==0:
                plt.plot(xx[:,0],xx[:,1],'bo')
            elif i==1: plt.plot(xx[:,0],xx[:,1],'rd')
            elif i==2: plt.plot(xx[:,0],xx[:,1],'gD')
            elif i==3: plt.plot(xx[:,0],xx[:,1],'c^')
            i+=1
        plt.show()




som = SOM()
som.loadDataSet('dataset2.txt')
som.train()
print(som.w)
som.showCluster(plt)


