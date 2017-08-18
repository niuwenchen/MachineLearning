#encoding:utf-8
#@Time : 2017/7/26 10:34
#@Author : JackNiu

import  numpy as np
X=[1,2]
state=[0.0,0.0]

# 分开定义不同输入部分的权重
w_cell_state = np.array([[0.1,0.2],[0.3,0.4]])
w_cell_input=np.array([0.5,0.6])
b_cell = np.array([0.1,-0.1])

# 定义输出的全连接参数
w_output=np.array([[1.0],[2.0]])
b_output=0.1

for i in range(len(X)):
    before_activation = np.dot(state,w_cell_input)+X[i]*w_cell_input+b_cell
    # state已经改变
    # np.tanh就是 (h+x)*Whh+bh==> Whh*ht-1+ Whh*Xt + bh,Whh 和Whx有同 有不同
    state = np.tanh(before_activation)
    final_output = np.dot(state,w_output)+b_output
    print("before activation:",before_activation)
    print("state:", state)
    print("output:",final_output)

