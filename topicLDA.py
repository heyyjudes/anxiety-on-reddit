import numpy as np
import gensim
import os

from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words

import feat
import svm
import NNet
import logreg

class LDA(feat.Feature):
    def __init__(self, name):
        self.diff_train = False
        self.train_targets = None
        self.dict_pos = None
        self.dict_neg = None
        self.corpus_pos = None
        self.corpus_neg = None
        self.dict_uni = None
        self.corpus_uni = None
        self.model_pos = None
        self.model_neg = None
        feat.Feature.__init__(self, name)

        return

    def add_text(self, train, train_labels, test, corpus=None):
        '''
        add text to feature model
        :param corpus: corpus txt of words with punctuation removed to build model
        :param train: training array with punctuation removed
        :param test: test array of posts with punctuation removed
        :return:
        '''
        self.train = train
        self.train_targets = train_labels
        self.test = test
        if corpus != None:
            self.corpus = corpus
            self.diff_train = True
        else:
            self.corpus = train
            self.diff_train = False
        return

    def clean_text(self):

        self.train = [z.lower().replace('\n', '').split() for z in self.train]
        self.test = [z.lower().replace('\n', '').split() for z in self.test]

        if self.diff_train == True:
            self.corpus = [z.lower().replace('\n', '').split() for z in self.corpus]
        else:
            self.corpus = self.train

        return

    def remove_stop(self):
        en_stop = get_stop_words('en')
        en_stop.append('just')
        temp = []
        for token in self.test:
            s_token = [i for i in token if not i in en_stop]
            temp.append(s_token)
        self.corpus = temp

        temp = []
        for token in self.train:
            s_token = [i for i in token if not i in en_stop]
            temp.append(s_token)
        self.corpus = temp

        if self.diff_train == True:
            temp = []
            for token in self.corpus:
                s_token = [i for i in token if not i in en_stop]
                temp.append(s_token)
            self.corpus = temp
        else:
            self.corpus = self.train
        return

    def stem_LDA(self):
        p_stemmer = PorterStemmer()

        temp = []
        for s_token in self.test:
            text = [p_stemmer.stem(s) for s in s_token]
            temp.append(text)
        self.test = temp

        temp = []
        for s_token in self.train:
            text = [p_stemmer.stem(s) for s in s_token]
            temp.append(text)
        self.train = temp

        if self.diff_train == True:
            temp = []
            for s_token in self.corpus:
                text = [p_stemmer.stem(s) for s in s_token]
                temp.append(text)
            self.corpus = temp
        else:
            self.corpus = self.train
        return

    def doc_term_matrix(self):
        # construct document term matrix
        # assign frequencies
        # case 1: our corpus is just our training examples
        # we build two topic modeling spaces
        if self.diff_train == False:
            neg_tokens = []
            pos_tokens = []
            for i in range(0, len(self.train)):
                #here is the negative class
                if self.train_targets[i] == 0:
                    neg_tokens.append(self.train[i])
                else:
                    pos_tokens.append(self.train[i])

            self.dict_pos = gensim.corpora.Dictionary(pos_tokens)
            self.dict_neg = gensim.corpora.Dictionary(neg_tokens)

            # converted to bag of words
            self.corpus_pos = [self.dict_pos.doc2bow(text) for text in pos_tokens]
            self.corpus_neg = [self.dict_neg.doc2bow(text) for text in neg_tokens]

        #case 2: we want to use unified model
        else:
            comb = self.train + self.corpus
            self.dict_uni = gensim.corpora.Dictionary(comb)
            self.corpus_uni = [self.dict_uni.doc2bow(text) for text in comb]

        return

    def prep_model(self):
        model_name = self.name
        print "cleaning"
        self.clean_text()
        print "removing stop"
        self.remove_stop()
        print "building model"
        print "stemming"
        self.stem_LDA()
        print "matrix"
        self.doc_term_matrix()
        print "creating model"

        if self.diff_train == False:
            if os.path.isfile('pos_' + model_name) and os.path.isfile('neg_' + model_name):
                self.model_pos = gensim.models.ldamodel.LdaModel.load('pos_' + model_name)
                self.model_neg = gensim.models.ldamodel.LdaModel.load('neg_' + model_name)
            else:
                self.create_model(model_name)
        else:
            if os.path.isfile('uni_' + model_name):
                self.model = gensim.models.ldamodel.LdaModel.load('uni_' + model_name)

            else:
                self.create_model(model_name)
        return

    def create_model(self, mod_label):
        if self.diff_train == False:
            self.model_pos = gensim.models.ldamodel.LdaModel(self.corpus_pos, num_topics=10, id2word=self.dict_pos, passes=20)
            self.model_pos.save('pos_' + mod_label)
            self.model_neg = gensim.models.ldamodel.LdaModel(self.corpus_neg, num_topics=10, id2word=self.dict_neg, passes=20)
            self.model_neg.save('neg_' + mod_label)
        else:
            self.model = gensim.models.ldamodel.LdaModel(self.corpus_uni, num_topics=20, id2word=self.dict_uni,
                                                             passes=20)
            self.model.save('uni_' + mod_label)
        return

    def buildWordVector(self, text):
        size = 10
        # same training as model so we have two spaces
        if self.diff_train == False:
            pos_vec = np.zeros(size).reshape((1, size))
            neg_vec = np.zeros(size).reshape((1, size))
            query_neg = self.dict_neg.doc2bow(text)
            query_pos = self.dict_pos.doc2bow(text)

            neg_tup = self.model_neg[query_neg]
            pos_tup = self.model_pos[query_pos]
            for (topic, prob) in neg_tup:
                neg_vec[0][topic] = prob

            for(topic, prob) in pos_tup:
                pos_vec[0][topic] = prob

            vec = np.concatenate((pos_vec, neg_vec), axis=1)

        else:
            vec = np.zeros(size*2).reshape((1, size*2))
            query_vec = self.dict_uni.doc2bow(text)
            try:
                vec_tup = self.model[query_vec]
                for (topic, prob) in vec_tup:
                    vec[0][topic] = prob
            except:
                pass
        return vec


if __name__ == "__main__":

    print('a. fetching data')
    with open('data/anxiety_content.txt', 'r') as infile:
        anxiety_posts = infile.readlines()

    with open('data/mixed_content.txt', 'r') as infile:
        mixed_posts = infile.readlines()

    with open('data/unlabeled_tweet.txt', 'r') as infile:
        unlabeled_posts = infile.readlines()

    y = np.concatenate((np.ones(len(mixed_posts)), np.zeros(len(anxiety_posts))))
    x = np.concatenate((mixed_posts, anxiety_posts))


    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.10)
    rs.get_n_splits(x)
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # x_train = x_train[:100]
        # x_test = x_test[:100]
        # y_train = y_train[:100]
        # y_test = y_test[:100]

        lda_feat = LDA('twt.lda')
        lda_feat.add_text(x_train, y_train, x_test, corpus=unlabeled_posts)
        lda_feat.prep_model()

        print('d. scaling')
        train_vecs = np.concatenate([lda_feat.buildWordVector(z) for z in lda_feat.train])
        train_vecs = scale(train_vecs)
        # Build test tweet vectors then scale
        test_vecs = np.concatenate([lda_feat.buildWordVector(z) for z in lda_feat.test])
        test_vecs = scale(test_vecs)

        print('e. logistical regression')
        # Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
        logreg.run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('f. svm ')
        svm.train_svm(train_vecs, test_vecs, y_train, y_test)

        print('Simple NN')
        NNet.simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)

        split += 1

    print('done')

