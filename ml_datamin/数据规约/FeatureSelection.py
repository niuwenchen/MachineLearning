#encoding:utf-8
#@Time : 2017/9/12 16:20
#@Author : JackNiu

from sklearn.feature_selection import VarianceThreshold
import numpy as np

print("去除低方差特征")
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.4 * (1 - .4)))
print(sel.fit_transform(X))
print(np.var(X,axis=0))
# [ 0.13888889  0.22222222  0.25      ]


print("单变量的特征选择")
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
print(X[0])
X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
print(X_new.shape)
print(X_new[0])

# 递归特征消除
print("SelectfromModel")
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

print(X.shape)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
