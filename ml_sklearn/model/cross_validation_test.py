#encoding:utf-8
#@Time : 2017/8/16 17:40
#@Author : JackNiu

import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn import  datasets
from sklearn import  svm

'''
交叉验证: cross_val_score 在估计器和数据集上调用帮助函数
'''
from sklearn.model_selection import  cross_val_score
from sklearn import preprocessing

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_transformed = scaler.transform(X_train)
# print(X_train[0])
# print(X_train_transformed[0])
#
# clf = svm.SVC(kernel="rbf",C=1)
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import accuracy_score
# from sklearn import  cross_validation
#
# scoring=['precision_macro', 'recall_macro']
# predicted = cross_val_predict(clf,iris.data,iris.target,cv=10)
# accuracy_score(iris.target,predicted)
# print()
#
#
# clf = svm.SVC(kernel="rbf",C=1)
# scores = cross_val_score(clf,iris.data,iris.target,cv=5,scoring='accuracy')
# print(scores)
# print(scores.mean(),scores.std()*2)
# # score=clf.score(X_test,y_test)
# # print(score)


import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
data = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
label=np.array([0, 1, 0, 1])

for train, test in kf.split(data):
     print("%s %s" % (train, test))
     X_train, X_test, y_train, y_test = data[train], data[test], label[train], label[test]
     print(X_train)
     print(X_test)
     print(y_train)
     print(y_test)