import numpy as np
import csv
from svm import train_svm
from logreg import run_logreg
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit
import arff

def read_liwc_csv(input_file):
    with open(input_file) as csvfile:
        reader = csv.DictReader(csvfile)
        output_arr = []
        for row in reader:
            del row['Filename']
            results = []
            for val in row.values():
                results.append(float(val))
            output_arr.append(results)
            labels = row.keys()

    return labels, output_arr

def build_train_test(x, y, train_index, test_index):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_test = scale(x_test)
    x_train = scale(x_train)

    return x_train, x_test, y_train, y_test

    return
if __name__ == "__main__":
    labels, anx_liwc= read_liwc_csv('data//anxious_liwc.csv')
    labels, mixed_liwc = read_liwc_csv('data//mixed_liwc.csv')

    y = np.concatenate((np.ones(len(mixed_liwc)), np.zeros(len(anx_liwc))))
    x = np.concatenate((mixed_liwc, anx_liwc))

    rs = ShuffleSplit(n_splits=10, test_size=.10, random_state=0)
    rs.get_n_splits(x)
    split = 0
    for train_index, test_index in rs.split(x):
        print split
        x_train, x_test, y_train, y_test = build_train_test(x, y, train_index, test_index)
        train_w_labels = np.concatenate((x_train, y_train.reshape(len(x_train), 1)), axis=1)

        arff.dump('result.arff', train_w_labels, relation='liwc', names=labels)
        print('log reg ')
        run_logreg(x_train, x_test, y_train, y_test)

        print('svm')
        train_svm(x_train, x_test, y_train, y_test)

        split +=1
