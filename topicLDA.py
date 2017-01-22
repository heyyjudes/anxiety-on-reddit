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


def prep_model(process):
    process.tokenize_LDA()
    process.remove_stop_LDA()
    process.stem_LDA()

def create_model(process, mod_label):
    process.doc_term_matrix()
    ldamodel = gensim.models.ldamodel.LdaModel(process.corpus, num_topics=10, id2word=process.dictionary, passes=20)
    ldamodel.save(mod_label)
    ldamodel = gensim.models.LdaModel.load(mod_label)
    print(ldamodel.print_topics(num_topics=10, num_words=10))
    return ldamodel

def buildWordVector(text, dic_neg, dic_pos, model_neg, model_pos):
    size = 10
    pos_vec = np.zeros(size).reshape((1, size))
    neg_vec = np.zeros(size).reshape((1, size))
    query_neg = dic_neg.doc2bow(text)
    query_pos = dic_pos.doc2bow(text)

    neg_tup = model_neg[query_neg]
    pos_tup = model_pos[query_pos]
    for (topic, prob) in neg_tup:
        neg_vec[0][topic] = prob

    for(topic, prob) in pos_tup:
        pos_vec[0][topic] = prob

    vec = np.concatenate((pos_vec, neg_vec), axis=1)
    return vec

def run_logreg(train_vecs, test_vecs, y_train, y_test):
    LR = SGDClassifier(loss='log', penalty='l1')
    LR.fit(train_vecs, y_train)

    print 'Train Accuracy: %.3f' % LR.score(train_vecs, y_train)
    print 'Test Accuracy: %.3f'%LR.score(test_vecs, y_test)
    return LR

def show_graph(lr, test_vecs, y_test, split):
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
    save_str = 'ROC_' + str(split) + 'png'
    plt.savefig(save_str)


if __name__ == "__main__":
    #sys.path.extend(['C:\\Users\\heyyj\\PycharmProjects\\Reddit'])
    anxietyProcess = PreProcessorLDA('data/anxiety_content.txt')
    prep_model(anxietyProcess)
    anxiety_posts = anxietyProcess.stemmed_tokens

    mixedProcess = PreProcessorLDA('data/mixed_content.txt')
    prep_model(mixedProcess)
    mixed_posts = mixedProcess.stemmed_tokens

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
        for i in range(0, len(x_train)):
              if y_train[i] == 0:
                  anxietyProcess.train_tokens.append(x_train[i])
              else:
                  mixedProcess.train_tokens.append(x_train[i])

        anx_model = create_model(anxietyProcess, 'anxiety.lda')
        mixed_model = create_model(mixedProcess, 'allsub.lda')

        print('d. scaling')
        train_vecs = np.concatenate([buildWordVector(z, anxietyProcess.dictionary, mixedProcess.dictionary, anx_model, mixed_model) for z in x_train])
        train_vecs = scale(train_vecs)
        # Build test tweet vectors then scale
        test_vecs = np.concatenate([buildWordVector(z, anxietyProcess.dictionary, mixedProcess.dictionary, anx_model, mixed_model) for z in x_test])
        test_vecs = scale(test_vecs)

        print('e. logistical regression')
        # Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
        lr = run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('f. svm ')
        train_svm(train_vecs, test_vecs, y_train, y_test)

        print('Simple NN')
        simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)

        #print('g. Neural Network')
        #show_graph(lr, test_vecs, y_test, split)
        split += 1

    print('done')

    # tokenize document into atomic elements
