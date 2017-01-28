from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier

def run_logreg(train_vecs, test_vecs, y_train, y_test):
    LR = SGDClassifier(loss='log', penalty='l1')
    LR.fit(train_vecs, y_train)

    print 'Train Accuracy: %.3f' % LR.score(train_vecs, y_train)
    print 'Test Accuracy: %.3f'%LR.score(test_vecs, y_test)
    return LR

def run_LR(x, y):
    rs = ShuffleSplit(n_splits=5, test_size=.20)
    rs.get_n_splits(x)
    print 'Logistic Regression: '
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr = run_logreg(x_train, x_test, y_train, y_test)
        split +=1
    return lr