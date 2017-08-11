#encoding:utf-8
#@Time : 2017/7/30 14:17
#@Author : JackNiu

in_f = open('data.csv')
lines = in_f.readlines()
in_f.close()
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]

from sklearn.model_selection import train_test_split
x, y = zip(*dataset)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


import re

def remove_noise(document):
    noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
    clean_text = re.sub(noise_pattern, "", document)
    return clean_text.strip()

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    lowercase=True,     # lowercase the text
    analyzer='char_wb', # tokenise by character ngrams
    ngram_range=(1,2),  # use ngrams of size 1 and 2
    max_features=20,  # keep the most common 1000 ngrams
    preprocessor=remove_noise
)
vec.fit(x_train)
print(vec.vocabulary_)
#  {'my': 526, 'ot': 594, '茅e': 948, '谩n': 973, 'k': 443, '贸v': 983, 'hh': 368, '21': 74,
print(vec.transform(x_train).toarray()[0])
# [34  7  2  1  5  2  2  2  7  3  4  3  2  6  3  2  3  0  4  3]  这个就是count


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)


score=classifier.score(vec.transform(x_test), y_test)
print(score)


