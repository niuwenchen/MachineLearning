#encoding:utf-8
#@Time : 2017/6/3 15:22
#@Author : JackNiu
'''
数据离散化代码，
      1           2           3           4
A     0    0.178698    0.257724    0.351843
An  240  356.000000  281.000000   53.000000
即(0, 0.178698]有240个，(0.178698, 0.257724]有356个，依此类推
'''

import pandas as pd
from sklearn.cluster import KMeans

datafile='data.xls'
processedfile= 'tmp/data_processed.xls'
typelabel ={u'肝气郁结证型系数':'A', u'热毒蕴结证型系数':'B', u'冲任失调证型系数':'C', u'气血两虚证型系数':'D', u'脾胃虚弱证型系数':'E', u'肝肾阴虚证型系数':'F'}

k = 4 #需要进行的聚类类别数

data = pd.read_excel(datafile)
keys=list(typelabel.keys())
# df
result = pd.DataFrame()


if __name__=='__main__':
    for i in range(len(keys)):
        print(u'正在进行%s 的聚类' % keys[i])
        kmodel = KMeans(n_clusters=k, n_jobs=4)
        kmodel.fit(data[[keys[i]]].as_matrix())


        r1 = pd.DataFrame(kmodel.cluster_centers_, columns=[typelabel[keys[i]]])  # 聚类中心
        r2 = pd.Series(kmodel.labels_).value_counts()  # 分类统计

        r2 = pd.DataFrame(r2, columns=[typelabel[keys[i]] + 'n'])  # 转为DataFrame

        r = pd.concat([r1, r2], axis=1).sort(typelabel[keys[i]])  # 匹配聚类中心和类别数目
        print(r)
        r.index = [1, 2, 3, 4]
        r[typelabel[keys[i]]] = pd.rolling_mean(r[typelabel[keys[i]]], 2)  # rolling_mean 用来计算相邻2列的均值，作为边界点

        r[typelabel[keys[i]]][1] = 0.0  #将原来的聚类中心改为边界点
        # 0    0.178698    0.257724    0.351843  这样一个量化范围
        result = result.append(r.T)


    result = result.sort()
    result.to_excel(processedfile)



