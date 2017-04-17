import numpy as np
import feat
import nltk
import NNet
import svm
import logreg
from sklearn.model_selection import ShuffleSplit
from nltk.collocations import *

class GramModel():
    def __init__(self, name):
        self.name = name
        self.bi_freq = None
        self.bi_prob = None
        self.unigram = None
        self.length = 0

class Bigram(feat.Feature):

    def __init__(self, name):
        self.pos_model = None
        self.neg_model = None
        self.unlab_model = None
        self.std_model = None
        feat.Feature.__init__(self, name)
        return

    def build_all_models(self, pos_corp, neg_corp, unlabel_corp, std_corp):
        '''
        Input corpra built with nltk PlaintextCorpusReader
        :param pos_corp:
        :param neg_corp:
        :param unlabel_corp:
        :param std_corp:
        :return:
        '''
        pos_model = GramModel('pos')
        print "pos model"
        self.pos_model = self.build_gram(pos_model, pos_corp)

        neg_model = GramModel('neg')
        print "neg model"
        self.neg_model = self.build_gram(neg_model, neg_corp)

        unlabel_model = GramModel('twt')
        self.unlab_model = self.build_gram(unlabel_model, unlabel_corp)

        std_model = GramModel('wiki')
        self.std_model = self.build_gram(std_model, std_corp)

    def build_gram(self, model, corp):
        '''
        Buidiing model object from corpus with bigram freq and prob and unigram freq
        :param model:
        :param corp:
        :return:
        '''
        model.corp = corp
        model.bi_freq = nltk.ConditionalFreqDist(nltk.bigrams(corp.words()))
        model.bi_prob = nltk.ConditionalProbDist(model.bi_freq, nltk.MLEProbDist)
        #print nltk.FreqDist(nltk.bigrams(corp.words())).most_common(20)
        #bigram_measures = nltk.collocations.BigramAssocMeasures()
        #finder = BigramCollocationFinder.from_words(corp.words())
        #finder.apply_freq_filter(10)
        #print finder.nbest(bigram_measures.pmi, 10)

        model.unigram = nltk.FreqDist(corp.words())
        model.length = len(corp.words())
        #print model.unigram.most_common(20)
        return model


    def calc_prob(self, input_tokens, model):
        '''
        use conditional prob dist model to calculate probability tokens
        :param input_tokens:
        :param model:
        :return:
        '''
        #build unigram freq model here: we need class
        first_token = input_tokens[0]
        first_prob = float((model.unigram[first_token] + 1)/model.length)
        unigram_prob = 0
        #convert to log domain to avoid underflow
        if first_prob > 0:
            total_prob = np.log(first_prob)
        else:
            total_prob = 0
        unigram_prob = total_prob
        for i in range(1, len(input_tokens)):
            temp_prob = model.bi_prob[input_tokens[i-1]].prob(input_tokens[i])
            temp_prob_uni = float((model.unigram[input_tokens[i]] + 1)/model.length)
            if temp_prob > 0:
                total_prob += np.log(temp_prob)
            if temp_prob_uni > 0:
                unigram_prob += np.log(temp_prob_uni)
        return unigram_prob, total_prob

    def build_prob_vecs(self, input_arr):
        feat_arr = []
        for arr in input_arr:
            tokens = arr.split(" ")
            feat_vec = np.zeros((8,))
            feat_vec[0:2] = self.calc_prob(tokens, self.pos_model)
            feat_vec[2:4] = self.calc_prob(tokens, self.neg_model)
            feat_vec[4:6] = self.calc_prob(tokens, self.unlab_model)
            feat_vec[6:8] = self.calc_prob(tokens, self.std_model)
            feat_arr.append(feat_vec)
        return np.asarray(feat_arr)


def build_corp(file_name):
    '''
    return corpus from file using nltk Plain text corpus reader
    :param file_name:
    :return:
    '''
    corpus = nltk.corpus.reader.plaintext.PlaintextCorpusReader("data/", file_name)
    return corpus

if __name__ == "__main__":
    print('a. fetching data')
    with open('data/anxiety_content.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()

    with open('data/unlabeled_tweet.txt', 'r') as infile:
        unlabeled_posts = infile.readlines()

    new_arr = []
    for post in dep_posts:
        if len(post) > 5:
            new_arr.append(post)
    dep_posts = new_arr

    new_arr = []
    for post in reg_posts:

        if len(post) > 5:
            new_arr.append(post)
    reg_posts = new_arr

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))


    brown_corp = nltk.corpus.brown
    unlabeled_corp = build_corp("unlabeled_tweet.txt")

    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.20, random_state=0)
    rs.get_n_splits(x)
    split = 0

    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        with open('data/anx_train_set.txt', 'w') as anx:
            with open('data/all_train_set.txt', 'w') as all:
                for i in range(0, len(x_train)):
                    if y_train[i] == 0:
                        anx.write(x_train[i])
                    else:
                        all.write(x_train[i])

        new_ngram = Bigram('reg')
        print "building corpus"
        pos_corp = build_corp("all_train_set.txt")
        neg_corp = build_corp("anx_test_set.txt")

        new_ngram.build_all_models(pos_corp, neg_corp, unlabeled_corp, brown_corp)

        print "calculating train"

        train_vecs = new_ngram.build_prob_vecs(x_train)

        print "calculating test"

        test_vecs = new_ngram.build_prob_vecs(x_test)

        np.save('feat/test_bigram' + str(split), test_vecs)
        np.save('feat/train_bigram' + str(split), train_vecs)

        print('Logreg')
        logreg.run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('SVM')
        svm.train_svm(train_vecs, test_vecs, y_train, y_test)

        print('Simple NN')
        NNet.simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)
        split += 1



