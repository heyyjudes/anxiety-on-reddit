# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, DocvecsArray
from gensim.models import Doc2Vec

# numpy
import numpy

import matplotlib.pyplot as plt

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from NNet import simpleNN

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
        #self.train_size = 9800
        self.train_size = 1980
        #self.test_size = 2400
        self.test_size = 660

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


if __name__ == "__main__":
    sources = {'data/anx_test_set.txt': 'TEST_NEG', 'data/mixed_test_set.txt': 'TEST_POS', 'data/anx_train_set.txt': 'TRAIN_NEG',
               'data/mixed_train_set.txt': 'TRAIN_POS', 'data/unlabeled_content.txt': 'TRAIN_UNS'}

    print('1. labeling')
    sentences = LabeledLineSentence(sources)

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=8)

    output_sen = sentences.to_array()
    model.build_vocab(output_sen)

    print('2. training doc2vec')
    for epoch in range(10):
        model.train(sentences.sentences_perm())
        print epoch

    print('3. saving model')
    model.save('./reddit.d2v')


    print('4. loading model')
    model = Doc2Vec.load('./reddit.d2v')

    train_arrays = numpy.zeros((sentences.train_size, 300))
    train_labels = numpy.zeros(sentences.train_size)

    for i in range(sentences.train_size/2):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_arrays[sentences.train_size/2 + i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 1
        train_labels[sentences.train_size/2 + i] = 0

    test_arrays = numpy.zeros((sentences.test_size, 300))
    test_labels = numpy.zeros(sentences.test_size)

    for i in range(sentences.test_size/2):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[i] = model.docvecs[prefix_test_pos]
        test_arrays[sentences.test_size/2 + i] = model.docvecs[prefix_test_neg]
        test_labels[i] = 1
        test_labels[sentences.test_size/2 + i] = 0

    print('5. logistic regression')
    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)

    print('f. plotting')
    show_graph(classifier, test_arrays, test_labels)

    print('5. simple neural network')
    simpleNN(train_arrays, test_arrays, train_labels, test_labels, 0.01, 25, 100)

    print 'Test Accuracy: %.2f'%classifier.score(test_arrays, test_labels)