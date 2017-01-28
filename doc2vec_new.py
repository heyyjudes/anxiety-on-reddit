# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, DocvecsArray
from gensim.models import Doc2Vec
from sklearn.model_selection import ShuffleSplit
# numpy
import numpy as np

import matplotlib.pyplot as plt
# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

#from NNet import simpleNN
from svm import train_svm
from LIWC_classify import split_array, parse_vec

def show_graph(lr, test_vecs, y_test):
    pred_probas = lr.predict_proba(test_vecs)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')

    plt.show()
    plt.savefig('ROC_doc.png')


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        self.train_neg_size = 0
        self.train_pos_size = 0
        self.test_pos_size = 0
        self.test_neg_size = 0
        self.train_size = 10000
        #self.train_size = 368
        self.test_size = 1000
        #self.test_size = 125

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
                    if prefix =='TRAIN_NEG':
                        if (self.train_neg_size >= self.train_size):
                            break
                        self.train_neg_size += 1
                    if prefix == 'TRAIN_POS':
                        if (self.train_pos_size >= self.train_size):
                            break
                        self.train_pos_size += 1
                    if prefix == 'TEST_NEG':
                        if (self.test_neg_size >= self.test_size):
                            break
                        self.test_neg_size += 1
                    if prefix == 'TEST_POS':
                        if (self.test_pos_size >= self.test_size):
                            break
                        self.test_pos_size += 1
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

    def find_words(self, tag):
        for sent in self.sentences:
            if tag in sent.tags:
                return sent.words


def build_d2v_vecs(split, train_index, test_index, x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    with open('data/anx_test_set.txt', 'w') as anx:
        with open('data/all_test_set.txt', 'w') as all:
            for i in range(0, len(x_test)):
                if y_test[i] == 0:
                    anx.write(x_test[i])
                else:
                    all.write(x_test[i])
    with open('data/anx_train_set.txt', 'w') as anx:
        with open('data/all_train_set.txt', 'w') as all:
            for i in range(0, len(x_train)):
                if y_train[i] == 0:
                    anx.write(x_train[i])
                else:
                    all.write(x_train[i])

    sources = {'data/anx_test_set.txt': 'TEST_NEG', 'data/all_test_set.txt': 'TEST_POS',
               'data/anx_train_set.txt': 'TRAIN_NEG',
               'data/all_train_set.txt': 'TRAIN_POS'}

    print('1. labeling')
    sentences = LabeledLineSentence(sources)

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=8)

    output_sen = sentences.to_array()
    model.build_vocab(output_sen)

    # print('2. training doc2vec')
    # for epoch in range(10):
    #     model.train(sentences.sentences_perm())
    #     print epoch
    #
    # print('3. saving model')
    # model.save(str(split) + '_reddit.d2v')
    model = Doc2Vec.load('models/' + str(split) + '_reddit.d2v')
    train_size = 20000
    test_size = 2000

    train_arrays = np.zeros((train_size, 300))
    train_labels = np.zeros(train_size)

    for i in range(train_size/2):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_arrays[train_size/2 + i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 1
        train_labels[train_size/2 + i] = 0

    test_arrays = np.zeros((test_size, 300))
    test_labels = np.zeros(test_size)

    for i in range(test_size/2):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[i] = model.docvecs[prefix_test_pos]
        test_arrays[test_size/2 + i] = model.docvecs[prefix_test_neg]
        test_labels[i] = 1
        test_labels[test_size/2 + i] = 0

    return train_arrays, test_arrays, train_labels, test_labels


if __name__ == "__main__":
    print('a. fetching data')
    with open('data/anxiety_content.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()

    # new_arr = []
    # for post in reg_posts:
    #     if len(post) > 5:
    #         new_arr.append(post)
    # reg_posts = new_arr
    #
    # new_arr = []
    # for post in dep_posts:
    #     if len(post) > 5:
    #         new_arr.append(post)
    # dep_posts = new_arr



    # with open('data/liwc_anxious.txt', 'r') as infile:
    #     anx_liwc_posts = infile.readlines()
    #
    # with open('data/liwc_mixed.txt', 'r') as infile:
    #     reg_liwc_posts = infile.readlines()
    #
    # reg_liwc_posts = split_array(reg_liwc_posts[0])
    # anx_liwc_posts = split_array(anx_liwc_posts[0])
    #
    # y_liwc = np.concatenate((np.ones(len(reg_liwc_posts)), np.zeros(len(anx_liwc_posts))))
    # x_liwc = np.concatenate((reg_liwc_posts, anx_liwc_posts))

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))

    # x = x[:len(x_liwc)]
    # y = y[:len(y_liwc)]

    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.10, random_state=0)
    rs.get_n_splits(x)
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split


        train_arrays, test_arrays, train_labels, test_labels = build_d2v_vecs(split, train_index, test_index, x, y)

        #addingliwc

        # x_train_liwc, x_test_liwc = x_liwc[train_index], x_liwc[test_index]
        # y_train_liwc, y_test_liwc = y_liwc[train_index], y_liwc[test_index]
        #
        # train_vecs_liwc = np.concatenate([parse_vec(z) for z in x_train_liwc])
        # test_vecs_liwc = np.concatenate([parse_vec(z) for z in x_test_liwc])
        #
        # train_vecs_liwc = train_vecs_liwc[:len(train_arrays)]
        # test_vecs_liwc = test_vecs_liwc[:len(test_arrays)]
        #
        # train_arrays = np.concatenate((train_arrays, train_vecs_liwc), axis=1)
        # test_arrays = np.concatenate((test_arrays, test_vecs_liwc), axis=1)

        print('5. logistic regression')
        classifier = LogisticRegression()
        classifier.fit(train_arrays, train_labels)
        print 'Train Accuracy: %.3f' % classifier.score(train_arrays, train_labels)
        print 'Test Accuracy: %.3f' % classifier.score(test_arrays, test_labels)

        print('SVM')
        train_svm(train_arrays, test_arrays, train_labels, test_labels)

        print('Simple neural network')
        #simpleNN(train_arrays, test_arrays, train_labels, test_labels, 0.01, 100, 100)

        split +=1


