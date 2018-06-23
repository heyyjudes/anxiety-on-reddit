#Features form De Choudhury et al.

import feat
import csv
import nltk
import numpy as np
import NNet
import svm
import logreg
from sklearn.model_selection import ShuffleSplit

class DeCh(feat.Feature):
    def __init__(self, name):
        self.pos_liwc = None
        self.neg_liwc = None
        feat.Feature.__init__(self, name)
        return

    def load_liwc(self, pos_liwc, neg_liwc):
        '''
        input file names to be read as liwc vectors
        :param pos_liwc: positive group liwc vectors
        :param neg_liwc: control group liwc vectors
        :return:None
        '''
        self.pos_liwc = self.read_liwc_csv(pos_liwc)
        print "mix liwc", len(self.pos_liwc)
        self.neg_liwc = self.read_liwc_csv(neg_liwc)
        print "anx liwc", len(self.neg_liwc)

    def read_liwc_csv(self, input_file):
        '''
        process LIWC csv file
        :param input_file: csv containing liwc vectors
        :return:
        '''
        with open(input_file) as csvfile:
            reader = csv.DictReader(csvfile)
            output_arr = []
            for row in reader:
                del row['Filename']
                results = []
                for val in row.values():
                    results.append(float(val))
                output_arr.append(results)
        return output_arr

    def build_dic_anew(self):
        '''
        Build dictionary of Affective Norms for English Words (Warriner et al.)
        :return: dictionary
        '''
        input_file = csv.DictReader(open('ref/Ratings_Warriner_et_al.csv', 'r'))
        condensed_dict = {}
        porter = nltk.PorterStemmer()
        for row in input_file:
            keyword = porter.stem(row["Word"])
            condensed_dict[keyword] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]), float(row["D.Mean.Sum"]))
        return condensed_dict

    def calculate_score_anew(self, tokens, dict):
        '''
        Calculate ANEW scores
        :param tokens: word-tokenized list of posts
        :param dict: ANEW dict
        :return: average affect, average dom
        '''
        #only include activation/arousal and dominance
        anew_count = total_count = avg_aff = avg_dom = 0
        for word in tokens:
            if word in dict:
                val, aff, dom = dict[word]
                anew_count += 1
                avg_aff += aff
                avg_dom += dom
            total_count += 1
        if anew_count > 0:
            avg_aff = avg_aff / anew_count
            avg_dom = avg_dom / anew_count
        else:
            avg_aff = 0
            avg_dom = 0
        return avg_aff, avg_dom

    def calculate_lexicon(self, tokens, file_name):
        '''
        count lexicon what appear in file
        :param tokens: word-tokenized list of posts
        :param file_name: name of file where each word is new line
        :return: fraction of occurance
        '''
        medfile = open(file_name, 'r')
        medlist = medfile.readlines()
        medlist = [z.rstrip("\n") for z in medlist]

        med_count = 0
        for word in tokens:
            if word in medlist:
                med_count += 1

        return 1.0*med_count/len(tokens)

    def build_feat(self, train_set, train_ind):
        '''
        build normaized count lexicon
        :param train_set: tokenized list of posts
        :param train_ind: training set indices
        :return: feature feature vectors
        '''
        #initializing
        anew_dict = self.build_dic_anew()
        liwc = np.concatenate((self.pos_liwc, self.neg_liwc))

        assert(len(train_ind) == len(train_set))

        train_vec = []
        for i in range(0, len(train_set)):
            tokens = train_set[i].split(" ")
            feat = []
            #adding anew
            aff, dom = self.calculate_score_anew(tokens, anew_dict)
            feat.append(aff)
            feat.append(dom)

            #adding lexicon count
            med = self.calculate_lexicon(tokens, 'ref/druglist.txt')
            feat.append(med)

            #adding anx_lex
            a_lex = self.calculate_lexicon(tokens, 'ref/anxiety_lex.txt')
            feat.append(a_lex)

            #addliwc
            feat = np.concatenate((feat, liwc[train_ind[i]]))
            train_vec.append(feat)
        return np.asarray(train_vec)


if __name__ == "__main__":

    print('a. fetching data')
    with open('data/anxiety_filtered.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/control_filtered.txt', 'r') as infile:
        reg_posts = infile.readlines()

    print "mix text", len(reg_posts)
    print "anx text", len(dep_posts)

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))

    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.20, random_state=0)
    rs.get_n_splits(x)
    split = 0

    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        feat_model = DeCh("reg")
        feat_model.load_liwc('data/mixed_liwc2007.csv', 'data/anxiety_filtered2007.csv')

        print "calculating train"
        train_vecs = feat_model.build_feat(x_train, train_index)

        print "calculating test"

        test_vecs = feat_model.build_feat(x_test, test_index)

        np.save('feat/test_de' + str(split), test_vecs)
        np.save('feat/train_de' + str(split), train_vecs)

        print('Simple NN')
        NNet.simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)

        print('Logreg')
        logreg.run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('SVM')
        svm.train_svm(train_vecs, test_vecs, y_train, y_test)

        split += 1