#encoding:utf-8
#@Time : 2017/8/18 10:32
#@Author : JackNiu

from gensim import corpora

documents = ["Human machine interface for lab abc computer applications",
           "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# remove words that appear only ones
from collections import  defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1

texts = [[token for token in text if frequency[token] > 1]
          for text in texts]

# print(texts)
dictionary = corpora.Dictionary(texts)
dictionary.save('./deerwester.dict')
print(dictionary)
print(dictionary.token2id)

new_doc = "Human computer computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)    # [(0, 1), (2, 2)]

'''
主要将语料库转换为 对应dictinary的表示。
将上面语料库保存为矩阵
'''

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./deerwester.mm', corpus)  # store to disk, for later use
print(corpus)