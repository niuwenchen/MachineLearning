#encoding:utf-8
#@Time : 2017/8/16 20:10
#@Author : JackNiu

from sklearn.model_selection import LeaveOneOut,LeavePOut
X=[1,2,3,4]
loo=LeaveOneOut()
for train,test in loo.split(X):
    print("%s %s"%(train,test))

# [1 2 3] [0]
# [0 2 3] [1]
# [0 1 3] [2]
# [0 1 2] [3]
import numpy as np

from sklearn.model_selection import ShuffleSplit
X1=np.arange(5)
ss=ShuffleSplit(n_splits=3,test_size=0.1,random_state=0)
for train,test in ss.split(X1):
    print("%s  %s"%(train,test))