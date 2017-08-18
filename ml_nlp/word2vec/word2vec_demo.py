#encoding:utf-8
#@Time : 2017/8/18 11:12
#@Author : JackNiu

# import modules & set up logging
import gensim, logging,os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

# model.save('./mymodel')
new_model = gensim.models.Word2Vec.load('./mymodel')

# class MySentences(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
#
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname)):
#                 yield line.split()
#
#
# sentences = MySentences('./directory')  # a memory-friendly iterator
# model = gensim.models.Word2Vec(sentences)

