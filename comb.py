import sys
import numpy as np
import gensim
from preprocess import PreProcessorLDA
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from svm import train_svm
from NNet import simpleNN
from gsim_word2vec import build_vecs_w2v
from doc2vec_new import build_d2v_vecs
import os.path

def prep_model(process):
    process.tokenize_LDA()
    process.remove_stop_LDA()
    process.stem_LDA()

def create_model(process, mod_label):
    process.doc_term_matrix()
    if os.path.exists(mod_label):
        ldamodel = gensim.models.ldamodel.LdaModel.load(mod_label)
        ldamodel.update(process.corpus)
        print "loading prexisting lda"
    else:
        print "creating new lda"
        ldamodel = gensim.models.ldamodel.LdaModel(process.corpus, num_topics=10, id2word=process.dictionary, passes=20)
        ldamodel.save(mod_label)
    print(ldamodel.print_topics(num_topics=10, num_words=10))
    return ldamodel

def buildWordVector(text, dic_neg, dic_pos, model_neg, model_pos):
    size = 10
    pos_vec = np.zeros(size).reshape((1, size))
    neg_vec = np.zeros(size).reshape((1, size))
    query_neg = dic_neg.doc2bow(text)
    query_pos = dic_pos.doc2bow(text)

    try:
        neg_tup = model_neg[query_neg]
        for (topic, prob) in neg_tup:
            neg_vec[0][topic] = prob
    except:
        print "word not found in model"
        print text

    try:
        pos_tup = model_pos[query_pos]
        for (topic, prob) in pos_tup:
            pos_vec[0][topic] = prob
    except:
        print "word not found in model"
        print text

    vec = np.concatenate((pos_vec, neg_vec), axis=1)
    return vec

def run_logreg(train_vecs, test_vecs, y_train, y_test):
    LR = SGDClassifier(loss='log', penalty='l1')
    LR.fit(train_vecs, y_train)

    print 'Train Accuracy: %.3f' % LR.score(train_vecs, y_train)
    print 'Test Accuracy: %.3f'%LR.score(test_vecs, y_test)
    return LR

def parse_vec(in_vec):

    result_list = [float(x) for x in in_vec]
    result_list = np.array(result_list)
    result_list = result_list.reshape((1, 70))
    return result_list

def split_array(long_string):
    split_arr = []
    result_list = long_string.split('-')
    for i in xrange(0, len(result_list)-69, 69):
        sub_arr = result_list[i:i+69]
        end_val = result_list[i+69]
        temp_arr = end_val.split('.')
        if len(temp_arr) > 2:
            result_list[i+69]=temp_arr[1][-1] + '.' + temp_arr[2]
        sub_arr.append(temp_arr[0] + '.' + temp_arr[1][:-1])
        split_arr.append(sub_arr)
    return split_arr
if __name__ == "__main__":
    #sys.path.extend(['C:\\Users\\heyyj\\PycharmProjects\\Reddit'])

    with open('data/liwc_anxious.txt', 'r') as infile:
        anx_liwc_posts = infile.readlines()

    with open('data/liwc_mixed.txt', 'r') as infile:
        reg_liwc_posts = infile.readlines()

    reg_liwc_posts = split_array(reg_liwc_posts[0])
    anx_liwc_posts = split_array(anx_liwc_posts[0])

    y_liwc = np.concatenate((np.ones(len(reg_liwc_posts)), np.zeros(len(anx_liwc_posts))))
    x_liwc = np.concatenate((reg_liwc_posts, anx_liwc_posts))

    anxietyProcess = PreProcessorLDA('data/anxiety_content.txt')
    new_arr = []
    for post in anxietyProcess.texts:
        if len(post) > 5:
            new_arr.append(post)
    anxietyProcess.texts = new_arr
    prep_model(anxietyProcess)
    anxiety_posts = anxietyProcess.stemmed_tokens


    print len(anxiety_posts)

    mixedProcess = PreProcessorLDA('data/mixed_content.txt')
    prep_model(mixedProcess)
    mixed_posts = mixedProcess.stemmed_tokens

    y = np.concatenate((np.ones(len(mixed_posts)), np.zeros(len(anxiety_posts))))
    x = np.concatenate((mixed_posts, anxiety_posts))

    # DOC2Vec uncomment
    # with open('data/anxiety_content.txt', 'r') as infile:
    #     dep_posts = infile.readlines()
    #
    # with open('data/mixed_content.txt', 'r') as infile:
    #     reg_posts = infile.readlines()
    #
    # new_arr = []
    # for post in dep_posts:
    #     if len(post) > 5:
    #         new_arr.append(post)
    # dep_posts = new_arr
    #
    # y_doc = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    # x_doc = np.concatenate((reg_posts, dep_posts))

    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.10, random_state=0)
    rs.get_n_splits(x)
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for i in range(0, len(x_train)):
              if y_train[i] == 0:
                  anxietyProcess.train_tokens.append(x_train[i])
              else:
                  mixedProcess.train_tokens.append(x_train[i])

        anx_model = create_model(anxietyProcess, 'models/'+ str(split) + '_anxiety.lda')
        mixed_model = create_model(mixedProcess, 'models/'+ str(split) + '_allsub.lda')

        x_train_liwc, x_test_liwc = x_liwc[train_index], x_liwc[test_index]
        y_train_liwc, y_test_liwc = y_liwc[train_index], y_liwc[test_index]

        print('d. scaling')
        train_vecs = np.concatenate([buildWordVector(z, anxietyProcess.dictionary, mixedProcess.dictionary, anx_model, mixed_model) for z in x_train])
        train_vecs = scale(train_vecs)
        # Build test tweet vectors then scale
        test_vecs = np.concatenate([buildWordVector(z, anxietyProcess.dictionary, mixedProcess.dictionary, anx_model, mixed_model) for z in x_test])
        test_vecs = scale(test_vecs)


        print "building training set"
        train_vecs_liwc = np.concatenate([parse_vec(z) for z in x_train_liwc])
        print "building test set"
        test_vecs_liwc = np.concatenate([parse_vec(z) for z in x_test_liwc])

        #print('d1. adding word2vec')
        # w_train_vecs, w_test_vecs = build_vecs_w2v(split, x_train, x_test)

        # train_vecs = np.concatenate((train_vecs, w_train_vecs), axis=1)
        # test_vecs = np.concatenate((test_vecs, w_test_vecs), axis=1)

        #print('d1. adding word2vec')
        #d_train_vecs, d_test_vecs, y_train, y_test= build_d2v_vecs(split,train_index, test_index, x_doc, y_doc)
        # train_vecs = train_vecs[:len(d_train_vecs)]
        # test_vecs = test_vecs[:len(d_test_vecs)]

        print train_vecs.shape
        print test_vecs.shape
        print train_vecs_liwc.shape
        # print train_vecs.shape
        print test_vecs_liwc.shape
        # print test_vecs.shape
        train_vecs = np.concatenate((train_vecs, train_vecs_liwc), axis=1)
        test_vecs = np.concatenate((test_vecs, test_vecs_liwc), axis=1)



        # train_vecs = np.concatenate((train_vecs, train_vecs_liwc), axis=1)
        # test_vecs = np.concatenate((test_vecs, test_vecs_liwc), axis=1)


        print('e. logistical regression')
        # Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
        lr = run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('f. svm ')
        train_svm(train_vecs, test_vecs, y_train, y_test)

        print('Simple NN')
        simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)

        split += 1

    print('done')

    # tokenize document into atomic elements
