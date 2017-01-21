import gensim
from sklearn.cross_validation import train_test_split
import numpy as np
import random
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def cleanText(corpus):
    corpus = [z.split() for z in corpus]
    return corpus


def labelizeReviews(reviews, label_type):
    labeled = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labeled.append(LabeledSentence(v, [label]))
    return labeled


# Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

def nparr_to_sent(np_arr):
    sent_arr = []
    for i in range(np_arr.shape[0]):
        sent_arr.append(LabeledSentence(np_arr[i][0], np_arr[i][1]))
    return sent_arr

def train_dm(x_train, x_test):
    size = 400

    # instantiate our DM and DBOW models
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # build vocab over all reviews
    # model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
    # model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))

    x_union = x_train + x_test
    model_dm.build_vocab(x_union)
    model_dbow.build_vocab(x_union)

    # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    # all_train_reviews = np.concatenate((x_train, unsup_reviews))

    all_train_reviews = x_train
    for epoch in range(10):
        perm = np.random.permutation(len(all_train_reviews))
        temp_train = np.array(all_train_reviews)
        train_input = nparr_to_sent(temp_train[perm])
        model_dm.train(train_input)
        model_dbow.train(train_input)
        print epoch

    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)

    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

    # train over test set
    x_test = np.array(x_test)

    for epoch in range(10):
        perm = np.random.permutation(x_test.shape[0])
        train_input = nparr_to_sent(x_test[perm])
        model_dm.train(train_input)
        model_dbow.train(train_input)
        print epoch

    # Construct vectors for test reviews
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)

    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs, test_vecs


def run_logreg(train_vecs, test_vecs, y_train, y_test):
    LR = SGDClassifier(loss='log', penalty='l1')
    LR.fit(train_vecs, y_train)

    print 'Train Accuracy: %.2f' % LR.score(train_vecs, y_train)
    print 'Test Accuracy: %.2f'%LR.score(test_vecs, y_test)
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
    with open('data/anxietysub_content.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()

    n_dim = 300

    print('b. initializing')

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))


    LabeledSentence = gensim.models.doc2vec.LabeledSentence

    #use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((reg_posts, dep_posts)), y, test_size=0.2)

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    #unsup_reviews = cleanText(unsup_reviews)

    #Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    #We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    #a dummy index of the review.

    print('c. labeling')
    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    #unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    print('d. training')
    train_vecs, test_vecs = train_dm(x_train, x_test)

    print('e. logistical regression')
    #Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
    lr = run_logreg(train_vecs, test_vecs, y_train, y_test)

    print('f. plotting')
    show_graph(lr, test_vecs, y_test, 0)

