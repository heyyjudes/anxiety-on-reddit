# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, DocvecsArray
from gensim.models import Doc2Vec

# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression

from NNet import simpleNN


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        self.train_neg_size = 0
        self.train_pos_size = 0
        self.test_pos_size = 0
        self.test_neg_size = 0
        self.train_size = 9800
        self.test_size = 2400

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
        self.sentences_pos = []
        self.sentences_dep = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):

                    if prefix =='TRAIN_NEG':
                        if (self.train_neg_size >= self.train_size):
                            break
                        self.train_neg_size += 1
                        self.sentences_dep.append(
                            LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
                    if prefix == 'TRAIN_POS':
                        if (self.train_pos_size >= self.train_size):
                            break
                        self.sentences_pos.append(
                            LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
                        self.train_pos_size += 1
                    if prefix == 'TEST_NEG':
                        if (self.test_neg_size >= self.test_size):
                            break
                        self.sentences_dep.append(
                            LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
                        self.test_neg_size += 1
                    if prefix == 'TEST_POS':
                        if (self.test_pos_size >= self.test_size):
                            break
                        self.sentences_pos.append(
                            LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
                        self.test_pos_size += 1
                    else:
                        self.sentences_pos.append(
                            LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
                        self.sentences_dep.append(
                            LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))

        return self.sentences_dep, self.sentences_pos

    def sentences_perm(self):
        shuffle(self.sentences_pos)
        shuffle(self.sentences_dep)
        return self.sentences_dep, self.sentences_pos

    def find_words(self, tag):
        for sent in self.sentences_dep:
            if tag in sent.tags:
                return sent.words

        for sent in self.sentences_pos:
            if tag in sent.tags:
                return sent.words



if __name__ == "__main__":
    sources = {'data/dep_test_set.txt': 'TEST_NEG', 'data/mixed_test_set.txt': 'TEST_POS', 'data/dep_train_set.txt': 'TRAIN_NEG',
               'data/mixed_train_set.txt': 'TRAIN_POS', 'data/unlabeled_content.txt':'TRAIN_UNS'}

    print('1. labeling')
    sentences = LabeledLineSentence(sources)

    model_dep = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model_reg = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)

    dep_sentences, pos_sentences = sentences.to_array()

    # model_dep.build_vocab(dep_sentences)
    # model_reg.build_vocab(pos_sentences)
    #
    # print('2. training doc2vec')
    # for epoch in range(5):
    #     dep_sent, reg_sent = sentences.sentences_perm()
    #     model_dep.train(dep_sent)
    #     model_reg.train(reg_sent)
    #     print epoch
    #
    # print('3. saving model')
    # model_dep.save('./reddit_dep.d2v')
    # model_reg.save('./reddit_reg.d2v')

    print('4. loading model')
    model_dep = Doc2Vec.load('./reddit_dep.d2v')
    model_reg = Doc2Vec.load('./reddit_reg.d2v')

    train_arrays = numpy.zeros((sentences.train_size, 200))
    train_labels = numpy.zeros(sentences.train_size)

    for i in range(sentences.train_size/2):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = numpy.concatenate((model_reg.docvecs[prefix_train_pos], model_dep.docvecs[prefix_train_pos]), axis=1)
        train_arrays[sentences.train_size/2 + i] = numpy.concatenate((model_reg.docvecs[prefix_train_neg], model_dep.docvecs[prefix_train_neg]), axis=1)
        train_labels[i] = 1
        train_labels[sentences.train_size/2 + i] = 0

    test_arrays = numpy.zeros((sentences.test_size, 200))
    test_labels = numpy.zeros(sentences.test_size)

    for i in range(sentences.test_size/2):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[i] = numpy.concatenate((model_reg.docvecs[prefix_test_pos], model_dep.docvecs[prefix_test_pos]), axis = 1)
        test_arrays[sentences.test_size/2 + i] = numpy.concatenate((model_reg.docvecs[prefix_test_neg], model_dep.docvecs[prefix_test_neg]), axis=1)
        test_labels[i] = 1
        test_labels[sentences.test_size/2 + i] = 0

    print('5. logistic regression')
    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)
    print 'Test Accuracy: %.2f'%classifier.score(test_arrays, test_labels)

    print('5. simple neural network')
    simpleNN(train_arrays, test_arrays, train_labels, test_labels, 0.01, 10, 100)

