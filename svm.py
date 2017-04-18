from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_score, recall_score

def train_svm(x_train, x_test, y_train, y_test):
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)

    print 'Train Accuracy: %.3f' % clf.score(x_train, y_train)
    print 'Test Accuracy: %.3f'%clf.score(x_test, y_test)
    print 'Test Percision %.3f' %precision_score(y_test, y_test_pred)
    print 'Test Recall %.3f' %recall_score(y_test, y_test_pred)

def run_SVM(x, y):
    print 'SVM: '
    rs = ShuffleSplit(n_splits=5, test_size=.20)
    rs.get_n_splits(x)
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_svm(x_train, x_test, y_train, y_test)
        split += 1