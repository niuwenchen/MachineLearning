#encoding:utf-8
#@Time : 2017/6/14 10:02
#@Author : JackNiu

from numpy import *
import matplotlib.pyplot as plt

class  BPNet(object):
    def __init__(self):
        self.eb=0.01
        self.iterator=0
        self.eta=0.1        # 步长
        self.mc =0.3        # 动量因子,用于玩过调优
        self.maxiter = 300
        self.nHidden =4
        self.nOut=1

        self.errlist=[]
        self.dataMat=0
        self.classLabels =0
        self.nSampleNum =0
        self.nSampleDim =0

    def logistic(self,net):
        return 1.0/(1.0+exp(-net))

    def dlogit(self,net):           # 实际的输出与期望输出的误差值
        return multiply(net, (1.0 - net))

    def  errorfunc(self,inX):
        return sum(power(inX, 2)) * 0.5

    def normalize(self,dataMat):        #数据归一化
        [m, n] = shape(dataMat)
        for i in range(n - 1):
            dataMat[:, i] = (dataMat[:, i] - mean(dataMat[:, i])) / (std(dataMat[:, i]) + 1.0e-10)  # (x-xba)/标准差
        return dataMat

    def loadDataSet(self,filename):
        self.dataMat = []
        self.classLabels = []
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split()
            self.dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])      # x,y,1
            self.classLabels.append(int(lineArr[2]))                                # label
        self.dataMat = mat(self.dataMat)

        m, n = shape(self.dataMat)

        self.nSampleNum = m  # 样本数量 实际上 是2列，但是增加了一个偏置 变成了3列
        self.nSampleDim = n - 1  # 样本维度

    #增加新列， 行数不变， mergeMat=matrix1+matrix2
    def addcol(self,matrix1,matrix2):
        [m1, n1] = shape(matrix1)
        [m2, n2] = shape(matrix2)
        if m1 != m2:
            print("different rows,can not merge matrix")
            return
        mergMat = zeros((m1, n1 + n2))
        mergMat[:, 0:n1] = matrix1[:, 0:n1]
        mergMat[:, n1:(n1 + n2)] = matrix2[:, 0:n2]
        return mergMat

# 隐藏层初始化
    def init_hiddenWB(self):
        # 4*2 + 4*1 = 4*3

        self.hi_w =2.0*(random.rand(self.nHidden,self.nSampleDim)-0.5)  #
        self.hi_b = 2.0*(random.rand(self.nHidden,1)-0.5)
        # 4*3
        self.hi_wb = mat(self.addcol(mat(self.hi_w), mat(self.hi_b)))      # 才是权重值，

    # 输出层初始化
    def init_OutputWB(self):
        self.out_w = 2.0 * (random.rand(self.nOut, self.nHidden) - 0.5)
        self.out_b = 2.0 * (random.rand(self.nOut, 1) - 0.5)
        self.out_wb = mat(self.addcol(mat(self.out_w), mat(self.out_b)))


    def bpTrain(self):
        # dataMat  307*3  逆后  3*307
        SampleIn= self.dataMat.T
        expected = mat(self.classLabels)
        self.init_hiddenWB()
        self.init_OutputWB()
        dout_wbOld =0.0   #应该是输出层的
        dhi_wbOld=0.0       #应该是隐含层
        for i in range(self.maxiter):
            # 信号正向传播, 输入层到隐含层  4*3 * 3*307 = 4*307
            hi_input = self.hi_wb*SampleIn
            hi_output =self.logistic(hi_input)
            #  因为每一个输入层的节点都有一个计算  307*5
            # print(hi_output.T.shape,ones((self.nS)))
            hi2out = self.addcol(hi_output.T,ones((self.nSampleNum,1))).T  # 输出层的偏置

            # 从隐含层到输出层   1*5 * 5*307 = 1*307
            out_input = self.out_wb*hi2out
            out_output = self.logistic(out_input)

            # 误差计算
            err= expected-out_output
            sse= self.errorfunc(err)
            self.errlist.append(sse)
            if sse<=self.eb:
                self.iterator=i+1
                break

            # 误差信号反向传播
            DELTA = multiply(err,self.dlogit(out_output))  # DELTA 为输出层梯度
            delta = multiply(self.out_wb[:,:-1].T*DELTA,self.dlogit(hi_output))
            dout_wb = DELTA*hi2out.T        # 输出层权值微分
            dhi_wb = delta*SampleIn.T       #

            # 更新
            if  i==0:
                self.out_wb = self.out_wb + self.eta * dout_wb
                self.hi_wb = self.hi_wb + self.eta * dhi_wb
            else:
                self.out_wb = self.out_wb + (1.0- self.mc)*self.eta* dout_wb + self.mc* dout_wbOld
                self.hi_wb = self.hi_wb + (1.0 - self.mc) * self.eta * dhi_wb + self.mc * dhi_wbOld

            dout_wbOld = dout_wb
            dhi_wbOld = dhi_wb

    def BPClassfier(self,start,end,steps=30):
        x = linspace(start, end, steps)
        xx = mat(ones((steps, steps)))
        xx[:, 0:steps] = x
        yy = xx.T
        z = ones((len(xx), len(yy)))
        for i in range(len(xx)):
            for j in range(len(yy)):
                xi = []
                tauex = []
                tautemp = []

                mat(xi.append([xx[i, j], yy[i, j], 1]))
                # 隐含层输入

                hi_input = self.hi_wb * (mat(xi).T)
                hi_out = self.logistic(hi_input)
                taumrow, taucol = shape(hi_out) #4*1

                tauex = mat(ones((1, taumrow + 1))) #1*5

                tauex[:, 0:taumrow] = (hi_out.T)[:, 0:taumrow]

                out_input = self.out_wb * (mat(tauex).T)
                out = self.logistic(out_input)

                z[i, j] = out
        # x 是一行30列， z是30行30列

        return x, z


    def classfyLine(self,plt,x,z):
        plt.contour(x, x, z, 1, cmap='RdGy')
    def trendLine(self,plt,color='r'):
        X = linspace(0, self.maxiter, self.maxiter)
        Y = log2(self.errlist)
        plt.plot(X, Y, color)

    def drawClassScatter(self,plt):
        i = 0
        for mydata in self.dataMat:
            if self.classLabels[i] == 0:
                plt.scatter(mydata[0, 0], mydata[0, 1], c='blue', marker='o')
            else:
                plt.scatter(mydata[0, 0], mydata[0, 1], c='red', marker='s')
            i += 1



