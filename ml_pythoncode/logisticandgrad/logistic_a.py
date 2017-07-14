#encoding:utf-8
#@Time : 2017/7/14 10:28
#@Author : JackNiu

from ml_pythoncode.logisticandgrad.util.common_libs import *
# 1.导入数据
Input = file2matrix("testSet.txt","\t")
target = Input[:,-1] #获取分类标签列表
[m,n] = shape(Input)


# 绘制散点图
def drawScatterbyLabel(plt,Input):
    m,n=shape(Input)
    target=Input[:,-1]
    for i in range(m):
        print(target[i])
        if target[i]==1.0:
            print("x")
            plt.scatter(Input[i,0],Input[i,1],c='blue',marker='o')
        else:
            print("y")
            plt.scatter(Input[i, 0], Input[i, 1], c='red', marker='*')

# 3.构建b+x 系数矩阵：b这里默认为1
dataMat = buildMat(Input)
# print dataMat
# 4. 定义步长和迭代次数
alpha = 0.001 # 步长
steps = 500  # 迭代次数
weights = ones((n,1))# 初始化权重向量
weightlist = []

# 5. 主程序
for k in range(steps):
    gradient = dataMat*mat(weights) # 梯度
    output = logistic(gradient)  # logistic函数
    errors = target-output # 计算误差
    weights = weights + alpha*dataMat.T*errors
    weightlist.append(weights)

print(weights)

# 6. 绘制训练后超平面
# drawScatterbyLabel(plt,Input)
# X = np.linspace(-7,7,100)
#y=w*x+b: b:weights[0]/weights[2]; w:weights[1]/weights[2]
# Y = -(double(weights[0])+X*(double(weights[1])))/double(weights[2])
# plt.plot(X,Y)
# plt.show()

def classifier(testData,weights):
    prob = logistic(sum(testData*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def hyberLineTrend():
    X=np.linspace(-5,5,100)
    Ylist=[]
    lenw=len(weightlist)
    for indx in range(lenw):
        if indx%20 ==0:
            weight = weightlist[indx]
            Y = -(double(weight[0]) + X * (double(weight[1]))) / double(weight[2])
            plt.plot(X, Y)
            plt.annotate("hplane:" + str(indx), xy=(X[99], Y[99]))
    plt.show()

def jieju():
    fig = plt.figure()
    axes1 = plt.subplot(211)
    axes2 = plt.subplot(212)
    weightmat=mat(zeros((steps,n)))
    i=0
    for weight in weightlist:
        weightmat[i,:]=weight.T
        i+= 1

    X=linspace(0,steps,steps)

    axes1.plot(X[0:10], -weightmat[0:10, 0] / weightmat[0:10,2],color="blue",linewidth=1,linestyle="-")  # 截距
    axes2.plot(X[10:], -weightmat[10:, 0] / weightmat[10:, 2], color="red", linewidth=1, linestyle="-")  # 截距

    plt.show()

def xielv():
    fig = plt.figure()
    axes1 = plt.subplot(211)
    axes2 = plt.subplot(212)
    weightmat = mat(zeros((steps, n)))
    i = 0
    for weight in weightlist:
        weightmat[i, :] = weight.T
        i += 1

    X = linspace(0, steps, steps)
    axes1.plot(X[0:10],-weightmat[0:10,1]/weightmat[0:10,2],color='blue',linewidth=1,linestyle="-")
    axes2.plot(X[10:],-weightmat[10:,1]/weightmat[10:,2],color="red",linewidth=1,linestyle="-")
    plt.show()

def  weight():
    axes1 = plt.subplot(311)
    axes2 = plt.subplot(312)
    axes3 = plt.subplot(313)
    weightmat = mat(zeros((steps, n)))
    i = 0
    for weight in weightlist:
        weightmat[i, :] = weight.T
        i += 1
    X = linspace(0, steps, steps)
    # 输出前10个点的截距变化
    axes1.plot(X, weightmat[:, 0], color='blue', linewidth=1, linestyle="-")
    axes1.set_ylabel('weight[0]')
    axes2.plot(X, weightmat[:, 1], color='red', linewidth=1, linestyle="-")
    axes2.set_ylabel('weight[1]')
    axes3.plot(X, weightmat[:, 2], color='green', linewidth=1, linestyle="-")
    axes3.set_ylabel('weight[2]')
    plt.show()

weight()
