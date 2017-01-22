import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit
from svm import train_svm
from NNet import simpleNN

def parse_vec(in_vec):
    result_list = in_vec.split('-')
    result_list = [float(x) for x in result_list]
    return result_list

def run_logreg(train_vecs, test_vecs, y_train, y_test):
    LR = SGDClassifier(loss='log', penalty='l1')
    LR.fit(train_vecs, y_train)

    print 'Train Accuracy: %.3f' % LR.score(train_vecs, y_train)
    print 'Test Accuracy: %.3f'%LR.score(test_vecs, y_test)
    return LR

if __name__ == "__main__":

    with open('data/liwc_anxious.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data/liwc_mixed.txt', 'r') as infile:
        reg_posts = infile.readlines()

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

        train_vecs = np.concatenate([parse_vec(z) for z in x_train])
        test_vecs = np.concatenate([parse_vec(z) for z in x_test])

        train_vecs=scale(train_vecs)
        test_vecs=scale(test_vecs)

        print('e. logistical regression')
        #Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
        lr = run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('f. svm')
        #show_graph(lr, test_vecs, y_test, split)
        train_svm(train_vecs, test_vecs, y_train, y_test)
        split += 1

        print('Simple NN')
        #simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 100, 100)
