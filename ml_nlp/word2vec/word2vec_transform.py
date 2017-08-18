#encoding:utf-8
#@Time : 2017/8/18 10:45
#@Author : JackNiu

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('./deerwester.dict')
corpus = corpora.MmCorpus('./deerwester.mm')
print(dictionary)
for ll in corpus:
    print(ll)
# 将上一篇保存的语料库对应的dictionary取出

tfidf = models.TfidfModel(corpus)
print("TfIdf训练完毕，分别输出整个语料库的 idf 和 tf")
print(tfidf.idfs)
print(tfidf.dfs)

print("对整个语料库进行训练，输出tf-idf的乘积")
for i in corpus:
    print(tfidf[i])


# 现在语料库中的每篇文档都可以用一个vector 来表示，词--> 数字表示
corpus_tfidf = tfidf[corpus]
lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=2)
corpus_lsi = lsi[corpus_tfidf]
print("lsi model")
for doc in corpus_lsi:
    print(doc)

