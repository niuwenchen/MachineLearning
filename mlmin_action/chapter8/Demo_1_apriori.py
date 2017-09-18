#encoding:utf-8
#@Time : 2017/6/5 9:55
#@Author : JackNiu
# 又是自行编写的Apriori函数。这里我们用sklearn 提供的函数

import time

from mlmin_action.chapter8.apriori.apriori_1 import *

inputfile='apriori.txt'
data = pd.read_csv(inputfile,header=None,dtype=object)   # object  默认对象数据
start = time.clock()
print('转换原始数据至0-1 矩阵')
ct= lambda x: pd.Series(1,index=x[pd.notnull(x)])
b=map(ct,data.as_matrix()) # 用map执行上面的lambda

data= pd.DataFrame(list(b)).fillna(0)  # 实现矩阵转换，除了1外，其余为空，用0 表示

end=time.clock()

print('转换完毕，用时: %0.2f秒' %(end-start))

del  b
support=0.06
confidence = 0.75
ms='---'  # 用来区分不同元素
start= time.clock()
print('开始搜索关联规则....')
find_rule(data,support,confidence,ms)
end= time.clock()
print('转换完毕，用时: %0.2f秒' %(end-start))
