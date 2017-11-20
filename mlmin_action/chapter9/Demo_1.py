#encoding:utf-8
#@Time : 2017/6/8 9:22
#@Author : JackNiu

import pandas as pd

inputfile="moment.csv"
data = pd.read_csv(inputfile,encoding="gbk").as_matrix()

from random import shuffle

from sklearn.model_selection import train_test_split

print(data[:,2:].shape)
print(data[:,0])

x_train,x_test,y_train,y_test= train_test_split(
    data[:,2:],data[:,0],test_size=0.2,random_state=0
)

x_train =x_train*30
y_train = y_train.astype(int)
x_test = x_test*30
y_test = y_test.astype(int)

from sklearn import svm
model = svm.SVC()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
import pickle
pickle.dump(model,open('./tmp/svm.model','wb'))

# 混淆矩阵
output1="./tmp/train.xls"
output2='./tmp/test.xls'
from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train,model.predict(x_train))
cm_test  = metrics.confusion_matrix(y_test,model.predict(x_test))
print(cm_train)
print(cm_test)
pd.DataFrame(cm_train,index=range(1,6),columns=range(1,6)).to_excel(output1)
pd.DataFrame(cm_test,index=range(1,6),columns=range(1,6)).to_excel(output2)
