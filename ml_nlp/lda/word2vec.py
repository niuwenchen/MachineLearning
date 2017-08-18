#encoding:utf-8
#@Time : 2017/8/14 11:14
#@Author : JackNiu

# 引入 word2vec
from gensim.models import word2vec

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 引入数据集
raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

# 切分词汇
sentences= [s.split(" ") for s in raw_sentences]
print(sentences)
# 构建模型
# model = word2vec.Word2Vec(sentences, min_count=1)
model =word2vec.Word2Vec.load('./word2vec')
# model.save("./word2vec")
# 进行相关性比较

x=model.similarity('dogs','you')
print(model.most_similar("lazy"))
