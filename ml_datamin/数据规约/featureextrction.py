#encoding:utf-8
#@Time : 2017/9/12 16:57
#@Author : JackNiu
from sklearn.feature_extraction import DictVectorizer

print("DictVectorizer  转换为python 数组")
measurements = [
  {'city': 'Dubai', 'temperature': 33.},
   {'city': 'London', 'temperature': 12.},
   {'city': 'San Francisco', 'temperature': 18.},
]

# vec = DictVectorizer()
# vecarray =vec.fit_transform(measurements).toarray()
# print(vecarray)
# print(vec.get_feature_names())

pos_window = [
  {
       'word-2': 'the',
      'pos-2': 'DT',
       'word-1': 'cat',
      'pos-1': 'NN',
       'word+1': 'on',
       'pos+1': 'PP',
   },
    # in a real application one would extract many such dictionaries
 ]

vec = DictVectorizer()
pos_vec = vec.fit_transform(pos_window)
print(pos_vec.toarray())
print(vec.get_feature_names())


## text Feature extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
corpus = [
        "This is the first document",
        "Thos is the second second document",
        "And the third one",
        "Is this the first document"
        ]
X= vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names())


bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),
                      token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print(analyze('Bi-grams are cool!'))
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print(X_2)
feature_index = bigram_vectorizer.vocabulary_.get('is this')
print(X_2[:,feature_index])

print("编码")
import chardet
text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"
text2 = b"holdselig sind deine Ger\xfcche"
text3 = b"\xff\xfeA\x00u\x00f\x00 \x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 \x00d\x00e\x00s\x00 \x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 \x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 \x00t\x00r\x00a\x00g\x00 \x00i\x00c\x00h\x00 \x00d\x00i\x00c\x00h\x00 \x00f\x00o\x00r\x00t\x00"

print(chardet.detect(text1)["encoding"])
decoded =[x.decode(chardet.detect(x)['encoding']) for x in (text1,text2,text3)]
v= CountVectorizer().fit(decoded).vocabulary_
for term in v:
    print(term)


print("图像特征提取")
from sklearn.feature_extraction import  image
import numpy as np
one_image = np.arange(4*4*3).reshape((4,4,3))
patchers = image.extract_patches_2d(one_image,(2,2),max_patches=2,random_state=0)
print(patchers.shape)
print(patchers[:,:,:,0])

patches = image.extract_patches_2d(one_image, (2, 2))
print(patches.shape)
print(patches[4,:,:,0])

