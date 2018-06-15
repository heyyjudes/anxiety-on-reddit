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
        self.words = 0

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
        model.length = len(corp.words())
        model.words = len(set(corp.words()))
        model.bi_freq = nltk.ConditionalFreqDist(nltk.bigrams(corp.words()))
        model.bi_prob = nltk.ConditionalProbDist(model.bi_freq, nltk.LaplaceProbDist, bins=model.words)
        model.unigram = nltk.FreqDist(corp.words())

        return model


    def calc_prob(self, input_tokens, model):
        '''
        use conditional prob dist model to calculate probability tokens
        :param input_tokens:
        :param model:
        :return:
        '''
        #build unigram freq model here: we need class
        delta = 1
        first_token = input_tokens[0]
        first_prob = float((model.unigram[first_token] + delta))/(model.length + delta*model.words)
        unigram_prob = 0
        #convert to log domain to avoid underflow
        total_prob = np.log(first_prob)
        unigram_prob = total_prob
        for i in range(1, len(input_tokens)):
            temp_prob = model.bi_prob[input_tokens[i-1]].prob(input_tokens[i])
            temp_prob_uni = float(model.unigram[input_tokens[i]] + delta)/(model.length + delta*model.words)
            total_prob += np.log(temp_prob)
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


    def analysis(self, pos_text, neg_text, result_text):
        pos_corp = build_corp(pos_text)
        neg_corp = build_corp(neg_text)

        #unigram most common
        pos_corp.unigram = nltk.FreqDist(pos_corp.words())
        pos_top = pos_corp.unigram.most_common(200)
        pos_top = [x for (x, c) in pos_top]

        neg_corp.unigram = nltk.FreqDist(neg_corp.words())
        neg_top = neg_corp.unigram.most_common(200)
        neg_top = [x for (x, c) in neg_top]

        result_file = open(result_text, 'w')
        result_file.write("frequent anxiety unigrams\n")
        for word in pos_top:
            if word not in neg_top:
                result_file.write(word + "\n")

        result_file.write("frequent reg unigrams\n")
        for word in neg_top:
            if word not in pos_top:
                    result_file.write(word + "\n")

        #bigram freq
        pos_corp.bigram = nltk.FreqDist(nltk.bigrams(pos_corp.words()))
        pos_top_bi = pos_corp.bigram.most_common(200)
        pos_top_bi = [x+ '_' +y for ((x, y), c) in pos_top_bi]

        neg_corp.bigram = nltk.FreqDist(nltk.bigrams(neg_corp.words()))
        neg_top_bi = neg_corp.bigram.most_common(200)
        neg_top_bi = [x+ '_' +y for ((x, y), c) in neg_top_bi]

        result_file.write("frequent anxiety bigrams\n")
        for word in pos_top_bi:
            if word not in neg_top_bi:
                result_file.write(word + "\n")

        result_file.write("frequent reg unigrams\n")
        for word in neg_top_bi:
            if word not in pos_top_bi:
                result_file.write(word + "\n")

        #collocation bigram
        result_file.write("\n")
        result_file.write("\n")

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(pos_corp.words())
        finder.apply_freq_filter(100)
        bigrams = finder.nbest(bigram_measures.pmi, 30)
        neg_bi_co = [x + '_' + y for (x, y) in bigrams]

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(neg_corp.words())
        finder.apply_freq_filter(100)
        bigrams = finder.nbest(bigram_measures.pmi, 30)
        pos_bi_co = [x + '_' + y for (x, y) in bigrams]

        result_file.write("bigram collocations anx\n")
        for b in pos_bi_co:
            if b not in neg_bi_co:
                result_file.write(str(b))
                result_file.write("\n")

        result_file.write("bigram collocations reg\n")
        for b in neg_bi_co:
            if b not in pos_bi_co:
                result_file.write(str(b))
                result_file.write("\n")

        result_file.write("\n")
        result_file.write("\n")

        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(pos_corp.words())
        finder.apply_freq_filter(75)
        trigrams = finder.nbest(trigram_measures.pmi, 30)
        neg_tri_co = [x + '_' + y + '_' + z for (x, y, z) in trigrams]

        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(neg_corp.words())
        finder.apply_freq_filter(75)
        trigrams = finder.nbest(trigram_measures.pmi, 30)
        pos_tri_co = [x + '_' + y + '_' + z for (x, y, z) in trigrams]

        result_file.write("trigram collocations anx\n")
        for t in neg_tri_co:
            if t not in pos_tri_co:
                result_file.write(str(t))
                result_file.write("\n")

        result_file.write("trigram collocations reg\n")
        for t in pos_tri_co:
            if t not in neg_tri_co:
                result_file.write(str(t))
                result_file.write("\n")







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
    with open('data/anxiety_filtered.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()

    with open('data/unlabeled_tweet.txt', 'r') as infile:
        unlabeled_posts = infile.readlines()

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))

    #for analyzing corpus
    new_ngram = Bigram('reg')
    new_ngram.analysis('anxiety_content.txt', 'mixed_content.txt', 'data/gram_result_new.txt')

    brown_corp = nltk.corpus.brown
    unlabeled_corp = build_corp("unlabeled_tweet.txt")

    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.20, random_state=0)
    rs.get_n_splits(x)
    split = 0

    for train_index, test_index in rs.split(x):
        print len(train_index)
        print len(test_index)
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
        neg_corp = build_corp("anx_train_set.txt")

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



