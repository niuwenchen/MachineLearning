#encoding:utf-8
#@Time : 2017/8/18 1:10
#@Author : JackNiu

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer()
corpus = [
   'This is the first document This.',
    'This is the second second document.',
    'and the third one.',
    'Is this the first document?',
]

X=vectorizer.fit_transform(corpus)
print("词典")
print(vectorizer.vocabulary_)
# and
print("TF_IDF 中的TF: ")
print(X.toarray())

print("计算IDF")
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X.toarray())
print( tfidf.toarray())
print(transformer.idf_)

print("直接用TFIDF计算")
vect = TfidfVectorizer()
y=vect.fit_transform(corpus)
print(y.toarray())
print(vect.idf_)