#encoding:utf-8
#@Time : 2017/6/14 11:43
#@Author : JackNiu


import matplotlib.pyplot as plt
from ml_network.from_ml_program.BP import BPNet

bpnet =BPNet()
bpnet.loadDataSet("testSet2.txt")
bpnet.dataMat = bpnet.normalize(bpnet.dataMat)
# bpnet.drawClassScatter(plt)
# plt.show()
bpnet.bpTrain()
print(bpnet.out_wb)
print(bpnet.hi_wb)
#
x,z= bpnet.BPClassfier(-3,3)
print('z',z)
bpnet.classfyLine(plt,x,z)
plt.show()

bpnet.trendLine(plt)
plt.show()
