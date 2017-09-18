#encoding:utf-8
#@Time : 2017/6/12 17:13
#@Author : JackNiu

import pandas as pd
from sqlalchemy import create_engine


engine = create_engine('mysql+pymysql://root:0000@127.0.0.1:3307/test?charset=utf8')
sql = pd.read_sql("all_gzdata",engine,chunksize=100)

counts= [i['fullURLId'].value_counts() for i in sql]
counts = pd.concat(counts).groupby(level=0).sum()  # 合并统计结果，把相同的统计项合并，按index分组并求和
counts = counts.reset_index()  # 重新设置index，将原来的index作为counts的一列
counts.columns=['index','num']
counts['type']=counts['index'].str.extract('(\d{3})') #提取前3个数字作为类别ID
                                                        # 新增一列
counts_ = counts[['type','num']].groupby('type').sum()  # 按列合并
counts_.sort('num',ascending = False)
print(counts_)

def count107(i):
    j =i[['fullURL']][i['fullURLId'].str.contains('107')].copy()
    j['type']=None
    j['type'][j['fullURL'].str.contains('info/.+?/')]=u'知识首页'
    j['type'][j['fullURL'].str.contains('info/.+?/.+?')]=u'知识列表页'
    j['type'][j['fullURL'].str.contains('/d+?_*\d+?\.html')] = u'知识内容页'
    return j['type'].value_counts()

counts2=[count107(i) for i in sql]
counts2= pd.concat(counts2).groupby(level=0).sum()  # 合并统计结果


