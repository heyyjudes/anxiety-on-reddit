from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier


def cleanText(corpus):
    corpus = [z.lower().replace('\n', '').split() for z in corpus]
    return corpus


def w2v_init(reg_posts, dep_posts, n_dim):
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((reg_posts, dep_posts)), y, test_size=0.2)

    # Do some very minor text preprocessing

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    # Initialize model and build vocab
    reddit_w2v = Word2Vec(size=n_dim, min_count=10)
    reddit_w2v.build_vocab(x_train)
    return x_train, x_test, y_train, y_test, reddit_w2v

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

def run_logreg(train_vecs, y_train):
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)
    print 'Train Accuracy: %.2f' % lr.score(train_vecs, y_test)
    print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)
    return lr

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

if __name__ == "__main__":

    print('1. fetching data')
    with open('data/depression_content.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()

    n_dim = 300

    print('2. initializing')
    x_train, x_test, y_train, y_test, reddit_w2v = w2v_init(reg_posts, dep_posts, n_dim)

    print('3. training model')
    #Train the model over train_reviews (this may take several minutes)
    reddit_w2v.train(x_train)

    print('4. scaling')
    train_vecs = w2v_train_scale(n_dim, reddit_w2v, x_train, x_test)

    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim, reddit_w2v) for z in x_test])
    test_vecs = scale(test_vecs)

    print('5. logistical regression')
    #Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
    lr = run_logreg(train_vecs, y_train)

    print('6. plotting')
    show_graph(lr, test_vecs, y_test)



