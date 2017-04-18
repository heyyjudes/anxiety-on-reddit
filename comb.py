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

results = np.zeros((10, 9))

for train_index, test_index in rs.split(x):
    print "split", split

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]


    test_vecs_w = np.load('feat/test_w2v' + str(split) + '.npy')
    train_vecs_w = np.load('feat/train_w2v' + str(split) + '.npy')


    test_vecs_c = np.load('feat/test_liwc' + str(split) + '.npy')
    train_vecs_c = np.load('feat/train_liwc' + str(split) + '.npy')

    test_vecs_d = np.load('feat/test_d2v' + str(split) + '.npy')
    train_vecs_d = np.load('feat/train_d2v' + str(split) + '.npy')

    test_vecs_u = np.load('feat/test_unibigram' + str(split) + '.npy')
    train_vecs_u = np.load('feat/train_unibigram' + str(split) + '.npy')

    test_vecs_l = np.load('feat/test_lda' + str(split) + '.npy')
    train_vecs_l = np.load('feat/train_lda' + str(split) + '.npy')

    test_vecs = np.concatenate((test_vecs_w, test_vecs_l), axis=1)

    train_vecs = np.concatenate((train_vecs_w, train_vecs_l), axis=1)

    print('Logreg')
    acc, per, rec = logreg.run_logreg(train_vecs, test_vecs, y_train, y_test)

    results[split][0] = acc
    results[split][1] = per
    results[split][2] = rec

    print('SVM')
    acc, per, rec = svm.train_svm(train_vecs_l, test_vecs_l, y_train, y_test)
    results[split][3] = acc
    results[split][4] = per
    results[split][5] = rec

    print('Simple NN')
    acc, per, rec = NNet.simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 10, 100)
    results[split][6] = acc
    results[split][7] = per
    results[split][8] = rec

    split += 1

print results
