from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from NNet import simpleNN
from sklearn.linear_model import SGDClassifier
from svm import train_svm

def cleanText(corpus):
    corpus = [z.lower().replace('\n', '').split() for z in corpus]
    return corpus


#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def w2v_train_scale(n_dim, reddit_w2v, x_train, x_test):
    train_vecs = np.concatenate([buildWordVector(z, n_dim, reddit_w2v) for z in x_train])
    train_vecs = scale(train_vecs)

    # Train word2vec on test tweets
    reddit_w2v.train(x_test)
    return train_vecs

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

    print('a. fetching data')
    with open('data/anxiety_content.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()

    n_dim = 300

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))


    print('b. initializing')
    rs = ShuffleSplit(n_splits=10, test_size=.10)
    rs.get_n_splits(x)
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train = cleanText(x_train)
        x_test = cleanText(x_test)

        # Initialize model and build vocab
        reddit_w2v = Word2Vec(size=n_dim, min_count=10)
        reddit_w2v.build_vocab(x_train)

        print('c. training model')
        #Train the model over train_reviews (this may take several minutes)
        reddit_w2v.train(x_train)

        print('d. scaling')
        train_vecs = w2v_train_scale(n_dim, reddit_w2v, x_train, x_test)

        #Build test tweet vectors then scale
        test_vecs = np.concatenate([buildWordVector(z, n_dim, reddit_w2v) for z in x_test])
        test_vecs = scale(test_vecs)

        print('e. logistical regression')
        #Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
        lr = run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('f. svm')
        #show_graph(lr, test_vecs, y_test, split)
        train_svm(train_vecs, test_vecs, y_train, y_test)
        split += 1

        print('Simple NN')
        simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)




