# LogReg Util
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score


def run_logreg(train_vecs, test_vecs, y_train, y_test):
    '''
    run logistic regression from sklearn and print output
    :param train_vecs: feature vectors training set
    :param test_vecs: feature vectors test set
    :param y_train: labels training set
    :param y_test: labels test set
    :return: accuracy, percision and recall
    '''
    LR = SGDClassifier(loss='log', penalty='l1')
    LR.fit(train_vecs, y_train)

    y_test_pred = LR.predict(test_vecs)

    acc = LR.score(test_vecs, y_test)
    per = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    print 'Train Accuracy: %.3f' % LR.score(train_vecs, y_train)
    print 'Test Accuracy: %.3f' %acc
    print 'Test Percision %.3f' %per
    print 'Test Recall %.3f' %recall
    return acc, per, recall


def run_LR(x, y):
    '''
    run cross validated logistic regression
    :param x: feature vectors
    :param y: labels
    :return: None
    '''
    rs = ShuffleSplit(n_splits=5, test_size=.20)
    rs.get_n_splits(x)
    print 'Logistic Regression: '
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc, per, recall = run_logreg(x_train, x_test, y_train, y_test)
        split +=1
    return