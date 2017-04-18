import sys
import numpy as np
import gensim
import NNet
import svm
import logreg
from preprocess import PreProcessorLDA
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from svm import train_svm
from NNet import simpleNN

print('a. fetching data')
with open('data/anxiety_filtered.txt', 'r') as infile:
    dep_posts = infile.readlines()

with open('data/mixed_content.txt', 'r') as infile:
    reg_posts = infile.readlines()

y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
x = np.concatenate((reg_posts, dep_posts))


rs = ShuffleSplit(n_splits=10, test_size=.20, random_state=0)
rs.get_n_splits(x)
split = 0

for train_index, test_index in rs.split(x):
    print "split", split
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]


    test_vecs_w = np.load('feat/test_w2v' + str(split) + '.npy')
    train_vecs_w = np.load('feat/train_w2v' + str(split) + '.npy')


    test_vecs_l = np.load('feat/test_liwc' + str(split) + '.npy')
    train_vecs_l = np.load('feat/train_liwc' + str(split) + '.npy')

    test_vecs = np.concatenate((test_vecs_w, test_vecs_l), axis=1)

    train_vecs = np.concatenate((train_vecs_w, train_vecs_l), axis=1)

    print('Simple NN')
    NNet.simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 200, 100)

    print('Logreg')
    logreg.run_logreg(train_vecs, test_vecs, y_train, y_test)

    print('SVM')
    svm.train_svm(train_vecs, test_vecs, y_train, y_test)


    split += 1