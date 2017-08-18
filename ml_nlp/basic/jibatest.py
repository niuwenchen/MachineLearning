#encoding:utf-8
#@Time : 2017/7/30 1:20
#@Author : JackNiu
import jieba

seg_list=jieba.cut("他毕业于上海交通大学，在百度深度学习研究院进行研究学习")
print('/'.join(seg_list))

seg_list=jieba.cut_for_search("他毕业于上海交通大学，在百度深度学习研究院进行研究学习")
print('/'.join(seg_list))

print('/'.join(jieba.cut('如果放到旧字典中将出错。',HMM=False)))
# 如果/放到/旧/字典/中将/出错/。
# 将连在一起的分成两个词
jieba.suggest_freq(('中','将'),True)
print('/'.join(jieba.cut('如果放到旧字典中将出错。',HMM=False)))
#如果/放到/旧/字典/中/将/出错/。
jieba.add_word('旧字典')
print('/'.join(jieba.cut('如果放到旧字典中将出错。',HMM=False)))
# 如果/放到/旧字典/中/将/出错/。

import jieba.analyse as analyse
lines=open('21.txt',encoding='utf-8').read()
# allowPOS 指定词性，形容词或名词之类的
print("  ".join(analyse.extract_tags(lines,topK=20,withWeight=False,allowPOS=())))


print(" ".join(analyse.textrank(lines,topK=20,withWeight=False,allowPOS=('ns','n','vn','v')
                                )))

import jieba.posseg as pseg
words = pseg.cut('我爱自然语言处理')
for word,flag in words:
    print(word,flag)

jieba.del_word('今天天气')
# jieba.suggest_freq(("今天","天气"),True)
terms = jieba.cut('今天天气不错',HMM=False)
print('/'.join(terms))