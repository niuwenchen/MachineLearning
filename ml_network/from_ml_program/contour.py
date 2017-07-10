#encoding:utf-8
#@Time : 2017/7/6 17:45
#@Author : JackNiu
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
print(X[0])
print(Y[0])
Z = f(X, Y)
print(Z[0])
plt.contour(X,Y,Z)
# plt.contour(X, Y, Z, colors='black')
plt.show()
