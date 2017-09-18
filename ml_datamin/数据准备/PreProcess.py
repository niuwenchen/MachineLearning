#encoding:utf-8
#@Time : 2017/9/12 15:12
#@Author : JackNiu

from sklearn import preprocessing
import numpy as np

# 标准差标准化
X=np.array([[1,-1,2],[2,0,0],[0,1,-1]])
X_scaled =preprocessing.scale(X)

print(X_scaled)

# stabdardscaler 类, model模型
scaler = preprocessing.StandardScaler().fit(X)
print(scaler.mean_)
print(scaler.std_)
print(scaler.transform(X))


### minMaxScaler  MaxAbsScaler
print("MinMAxScaler")
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit(X)
print(x_train_minmax.min_)
print(x_train_minmax.data_max_)
print(x_train_minmax.transform(X))

# 规范化
X_normalized = preprocessing.normalize(X, norm='l2')
print(X_normalized)

# 二值化
print("二值化")
binarizer = preprocessing.Binarizer().fit(X)
print(binarizer.transform(X))

# 独热编码
print("独热编码")
enc= preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
print(enc.transform([[0,1,3]]).toarray())

# 缺失值
print("缺失值")
import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

# 多项式特征
print("生成多项式特征")
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
print(np.arange(5))
print(X)
poly = PolynomialFeatures(2)
print(poly.fit_transform(X))
