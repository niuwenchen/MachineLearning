#encoding:utf-8
#@Time : 2017/7/14 11:34
#@Author : JackNiu


from ml_pythoncode.logisticandgrad.util.common_libs import *


Input = file2matrix("testSet.txt","\t")
target = Input[:,-1] #获取分类标签列表
[m,n] = shape(Input)

# drawScatterbyLabel(plt,Input)

dataMat = buildMat(Input)

# 4. 定义迭代次数
steps = 500  # 迭代次数
weights = ones(n) # 初始化权重向量



# alpha变化
alphalist =[]
alphahlist =[]
#
# # 算法主程序:
# # 1.对数据集的每个行向量进行m次随机抽取
# # 2.对抽取之后的行向量应用动态步长
# # 3.进行梯度计算
# # 4.求得行向量的权值，合并为矩阵的权值
#
weightlist=[]
for j in range(steps):
    dataIndex=range(m)
    for i in range(m):
        alpha= 2/(1.0+j+i)+0.001   # 修改alpha
        if j == 0: alphalist.append(alpha)
        if i == 0: alphahlist.append(alpha)
        randIndex = int(random.uniform(0, len(dataIndex)))  # 生成0~m之间随机索引
        vectSum = sum(dataMat[randIndex] * weights.T)  # 计算dataMat随机索引与权重的点积和
        grad = logistic(vectSum)  # 计算点积和的梯度
        errors = target[randIndex] - grad  # 计算误差
        weights = weights + alpha * errors * dataMat[randIndex]  # 计算行向量权重
        # del (dataIndex[randIndex])      # 从数据集中删除选取的随机索引
    weightlist.append(weights)

# weights	= weights.tolist()[0]
# lenal=  len(alphalist); lenalh=  len(alphahlist)
# fig = plt.figure()
# axes1 = plt.subplot(211); axes2 = plt.subplot(212)
# X1 = np.linspace(0,lenal,lenal); X2 = np.linspace(0,lenalh,lenalh)
# axes1.plot(X1,alphalist); axes2.plot(X2,alphahlist)
# plt.show()


def  shoulian():
    lenwl = len(weightlist)
    weightmat = zeros((lenwl, n))
    i = 0
    for weight in weightlist:
        weightmat[i, :] = weight
        i += 1
    fig = plt.figure()
    axes1 = plt.subplot(211);
    axes2 = plt.subplot(212)
    X1 = np.linspace(0, lenwl, lenwl)
    axes1.plot(X1, -weightmat[:, 0] / weightmat[:, 2])  # 截距
    axes1.set_ylabel('Intercept')
    axes2.plot(X1, -weightmat[:, 1] / weightmat[:, 2])  # 斜率
    axes2.set_ylabel('Slope')
    # 生成回归线
    ws = standRegres(X1, -weightmat[:, 0] / weightmat[:, 2])
    Y1 = ws[0, 0] + X1 * ws[1, 0]
    axes1.plot(X1, Y1, color='red', linewidth=2, linestyle="-");
    plt.show()


def quanzhong():
    lenwl = len(weightlist)
    weightmat = zeros((lenwl, n))
    i = 0
    for weight in weightlist:
        weightmat[i, :] = weight
        i += 1
    fig = plt.figure()
    axes1 = plt.subplot(311)
    axes2 = plt.subplot(312)
    axes3 = plt.subplot(313)
    X1 = np.linspace(0, lenwl, lenwl)
    axes1.plot(X1, weightmat[:, 0]);  #
    axes1.set_ylabel('weight[0]')
    axes2.plot(X1, weightmat[:, 1]);  #
    axes2.set_ylabel('weight[1]')
    axes3.plot(X1, weightmat[:, 2]);  #
    axes3.set_ylabel('weight[2]')
    plt.show()

quanzhong()