KNN

计算已知类别数据集中的点与当前点之间的距离
按照距离递增次序排序
选取与当前点距离最小的k个点
确定前k个点所在类别的出现频率
返回前k个点出现频率最高的类别作为当前点的预测分类

 'dict' object has no attribute 'iteritems'
 
 sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
实际使用这个算法时，算法的执行效率并不高，因为算法需要为每个测试向量做2000次的距离计算，
每个距离计算包括了2014 个维度浮点运行，总计要执行900次，还需要为测试向量准备2M的存储空间

k决策树就是k-近邻算法的优化版。
