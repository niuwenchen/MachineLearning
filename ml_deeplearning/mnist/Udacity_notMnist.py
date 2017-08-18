#encoding:utf-8
#@Time : 2017/8/16 16:46
#@Author : JackNiu

import pickle

from sklearn.linear_model import LogisticRegressionCV


all_pickle='notMNIST.pickle'
with open(all_pickle,'rb') as fp:
    notmnist = pickle.load(fp)
    train_dataset= notmnist['train_dataset']
    train_label = notmnist['train_labels']
    test_dataset = notmnist['test_dataset']
    test_label = notmnist['test_labels']
    valid_dataset = notmnist['valid_dataset']
    valid_labels = notmnist['valid_labels']
    print(train_dataset.shape)


train_samples = train_dataset.shape[0]
print(train_samples)

clf = LogisticRegressionCV(
                         multi_class='multinomial',
                         penalty='l2', solver='sag', tol=0.1)
clf.fit(train_dataset, train_label)

score = clf.score(valid_dataset, valid_labels)
print(score)
print(clf.score(test_dataset,test_label))


