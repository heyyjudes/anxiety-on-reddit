from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score

def run_logreg(train_vecs, test_vecs, y_train, y_test):
    LR = SGDClassifier(loss='log', penalty='l1')
    LR.fit(train_vecs, y_train)

    y_test_pred = LR.predict(test_vecs)
    print 'Train Accuracy: %.3f' % LR.score(train_vecs, y_train)
    print 'Test Accuracy: %.3f' %LR.score(test_vecs, y_test)
    print 'Test Percision %.3f' %precision_score(y_test, y_test_pred)
    print 'Test Recall %.3f' %recall_score(y_test, y_test_pred)
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