## Word2vec
Word2vec: 将词表征为实数值向量的高效工具，采用的模型有CBOW(Continuous Bag-of-Words，连续的词袋模型)和
Skip-Gram两种。

Word2vec一般被外界认为是一个Deep Learning（深度学习）的模型，可能word2vec是一种神经网络模型相关，但是认为该
模型层次较浅，严格的来说还不能算是深层模型。当然如果word2vec上层再套一层与具体应用相关的输出层，比如Softmax，
此时更像一个深层模型。

word2vec经过训练，可以把对文本呃逆荣的处理简化为K维向量空间中的向量运算，而向量空间上的相似度可以用来表示文本语义上的相似度。
因此，word2vec输出的词向量可以被用该做很多NLP相关的工作，比如聚类、找同义词、词性分析等等。而word2vec被人为认可的地方是
其向量的加法组合运算(Additive Compositionality): 

    vector('Pairs') - vector('France')  + vector('ltaly') = vector('Rome')
    vector('king')- vector('man')+vector('woman') =vector('queen')
    
词向量

    1 One-hot Representation
        NLP相关任务中最常见的第一步就是创建一个词表库并把每个词顺序编号。这实际就是词表示方法中的One-hot repres..
        这种方法把每个词顺序编号，每个词就是一个很长的向量，向量的温度等于词表大小，只有对应位置上的数字为1，其他都为0.
        当然，在实际应用中，一般次啊用稀疏编码存储，主要采用词的编号。
        这种表示方法最大的问题是无法捕捉词与词之间的相似度，就算是近义词也无法从词向量中看出任何关系。此外这种表示方法还容易
        引发维数灾难，尤其是在DL相关的一些应用中。
        
    2 Distributed Representation
        基本思想是通过训练将每个词映射成K维实数向量(K一般为模型中的超参数)，通过词之间的距离(cosine相似度，欧式距离)来
        判断它们之间的语义相似度。而word2vec就是这种Distributed Representation 的词向量表示方法。
        
## 统计语言模型
传统的统计语言模型表示语言基本单位(一般为句子)的概率分布函数，也就是说，语言模型就是概率分布函数，这个概率分
布也就是该语言的生成模型。


