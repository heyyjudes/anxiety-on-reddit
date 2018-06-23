import numpy as np
import os
import csv
import feat
import NNet
import svm
import logreg
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit

class W2V(feat.Feature):
    def __init__(self, name, dim, liwc):
        self.train = []
        self.test = []
        self.corpus = []
        self.dim = dim
        if liwc:
            self.build_liwc = True
        else:
            self.build_liwc = False
        self.liwc = None
        feat.Feature.__init__(self, name)
        return

    def add_text(self, corpus, train, test):
        '''
        add text to feature model
        :param corpus: corpus txt of words with punctuation removed to build model
        :param train: training array with punctuation removed
        :param test: test array of posts with punctuation removed
        :return:
        '''
        self.corpus = corpus
        self.train = train
        self.test = test
        return

    def load_liwc(self, pos_liwc, neg_liwc):
        '''
        input file names to be read as liwc vectors
        :param pos_liwc:
        :param neg_liwc:
        :return:
        '''
        pos_liwc = self.read_liwc_csv(pos_liwc)
        neg_liwc = self.read_liwc_csv(neg_liwc)
        self.liwc = np.concatenate((pos_liwc, neg_liwc))

    def read_liwc_csv(self, input_file):
        with open(input_file) as csvfile:
            reader = csv.DictReader(csvfile)
            output_arr = []
            for row in reader:
                del row['Filename']
                del row['Segment']
                results = []
                for val in row.values():
                    results.append(float(val))
                output_arr.append(results)
        return output_arr

    def clean_text(self):
        '''

        :param corpus: array of sentence strings
        :return:  remove \n items from sentences
        '''
        self.corpus = [z.lower().replace('\n', '').split() for z in self.corpus]
        self.train = [z.lower().replace('\n', '').split() for z in self.train]
        self.test = [z.lower().replace('\n', '').split() for z in self.test]
        return

    def build_vector(self, text):
        '''
        usiing learned vectors to calculate average embedding of a post
        :param text:
        :return:
        '''
        vec = np.zeros(self.dim).reshape((1, self.dim))
        count = 0.
        for word in text:
            try:
                vec += self.model[word].reshape((1, self.dim))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    def build_scale(self, input_vecs):
        '''
        clacluating embeddings for all posts and scaling embeddings
        :param input_vecs: array of array of word tokens
        :return:
        '''
        train_vecs = np.concatenate([self.build_vector(z) for z in input_vecs])
        train_vecs = scale(train_vecs)
        return train_vecs

    def build_model(self, model_vecs):
        '''
        building model if model not found
        set self.model to found model or newly built model
        :param model_vecs: input corpus string for training model
        :return:
        '''
        model_str = 'models/' + self.name + '_model.w2v'
        if os.path.isfile(model_str):
            model = Word2Vec.load(model_str)
        else:
            model = Word2Vec(size=self.dim, min_count=10)
            model.build_vocab(model_vecs)
            model.train(model_vecs)
            model.save('models/' + self.name + '_model.w2v')
        self.model = model
        return model

    def train_build_vectors(self, model_vecs, x_train, x_test, train_ind, test_ind):
        '''
        loading text, building model and
        :param model_vecs:
        :param x_train:
        :param x_test:
        :return:
        '''
        self.add_text(model_vecs, x_train, x_test)
        self.clean_text()
        self.build_model(self.corpus)

        self.train_feat_vecs = self.build_scale(self.train)
        self.test_feat_vecs = self.build_scale(self.test)

        if self.build_liwc:
            liwc_train_vec = self.liwc[train_ind]
            liwc_test_vec = self.liwc[test_ind]

            assert(len(liwc_test_vec) == len(self.test_feat_vecs))
            assert(len(liwc_train_vec) == len(self.train_feat_vecs))

            self.train_feat_vecs = np.concatenate((self.train_feat_vecs, liwc_train_vec), axis=1)
            self.test_feat_vecs = np.concatenate((self.test_feat_vecs, liwc_test_vec), axis=1)

        return self.train_feat_vecs, self.test_feat_vecs

if __name__ == "__main__":
    print('a. fetching data')
    with open('data/anxiety_filtered.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/control_filtered.txt', 'r') as infile:
        reg_posts = infile.readlines()

    with open('data/unlabeled_tweet.txt', 'r') as infile:
        unlabeled_posts = infile.readlines()

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))

    print len(y)
    print len(x)
    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.10, random_state=0)
    rs.get_n_splits(x)
    split = 0

    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        new_w2v = W2V('w2v' + str(split), 300, liwc=True)

        new_w2v.load_liwc('data/mixed_liwc.csv', 'data/anxious_liwc.csv')
        train_vecs, test_vecs = new_w2v.train_build_vectors(x_train, x_train, x_test, train_index, test_index)
        print len(train_vecs)
        print len(test_vecs)
        #trained with twitter corpus vectors
        #new_w2v = W2V('w2v_twt', 300, liwc=False)
        #train_vecs, test_vecs = new_w2v.train_build_vectors(unlabeled_posts, x_train, x_test, train_index, test_index)

        np.save('feat/test_w2v_liwc' + str(split), test_vecs)
        np.save('feat/train_w2v_liwc' + str(split), train_vecs)
        print('Simple NN')
        NNet.simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)

        print('Logreg')
        logreg.run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('SVM')
        svm.train_svm(train_vecs, test_vecs, y_train, y_test)

        split += 1





