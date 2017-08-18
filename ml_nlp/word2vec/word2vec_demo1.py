#encoding:utf-8
#@Time : 2017/8/18 11:48
#@Author : JackNiu

# 引入 word2vec
from gensim.models import word2vec

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 引入数据集
raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

# 切分词汇
sentences= [s.split() for s in raw_sentences]

# 构建模型
model = word2vec.Word2Vec(sentences, min_count=1,size=20)
print(model['you'])
print(model.compute_loss)
'''
直接获取某个单词的向量表示，也就是说该单词可以用别的10个单词来描述
[-0.02901401 -0.04214518 -0.02739167  0.04414326  0.01985594  0.02784069
 -0.0034573   0.049661    0.02452401  0.01624114]

'''

print(model['lazy'])
print(model.most_similar(positive=["dogs"],topn=30))
# 进行相关性比较
sim =model.similarity('dogs','you')
print(sim)

'''
具体的里面的模型解释还需要后面的知识来进行分析。

'''