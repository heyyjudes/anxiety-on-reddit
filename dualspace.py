from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from NNet import simpleNN
from sklearn.linear_model import SGDClassifier


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


def w2v_train_scale(n_dim, reddit_w2v, x_train):
    train_vecs = np.concatenate([buildWordVector(z, n_dim, reddit_w2v) for z in x_train])

    #train_vecs = scale(train_vecs)
    # Train word2vec on test tweets
    #reddit_w2v.train(x_test)
    return train_vecs

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
        print(len(dep_posts))

    with open('data/mixed_content.txt', 'r') as infile:
        reg_posts = infile.readlines()
        print(len(reg_posts))
    n_dim = 150

    y_reg = np.ones(len(reg_posts))
    y_dep = np.zeros(len(dep_posts))
    if len(reg_posts) > len(dep_posts):
        del reg_posts[len(dep_posts):]
    else:
        del dep_posts[len(reg_posts):]

    print(len(dep_posts))
    print(len(reg_posts))
    #x_reg = np.concatenate((reg_posts, dep_posts))

    y_reg = np.ones(len(reg_posts))
    y_dep = np.zeros(len(dep_posts))
    dep_posts = np.array(dep_posts)
    reg_posts = np.array(reg_posts)

    print('b. initializing')
    rs = ShuffleSplit(n_splits=2, test_size=.20)
    rs.get_n_splits(reg_posts)
    split = 0
    for train_index, test_index in rs.split(reg_posts):
        print "split", split
        x_train_dep, x_test_dep = dep_posts[train_index], dep_posts[test_index]
        x_train_reg, x_test_reg = reg_posts[train_index], reg_posts[test_index]

        y_train_dep, y_test_dep = y_dep[train_index], y_dep[test_index]
        y_train_reg, y_test_reg = y_reg[train_index], y_reg[test_index]

        y_train = np.concatenate((y_train_dep, y_train_reg))
        y_test = np.concatenate((y_test_dep, y_test_reg))

        x_train_dep = cleanText(x_train_dep)
        x_test_dep = cleanText(x_test_dep)

        x_train_reg = cleanText(x_train_reg)
        x_test_reg = cleanText(x_test_reg)

        #Initialize model and build vocab
        reddit_w2v_dep = Word2Vec(size=n_dim, min_count=10)
        reddit_w2v_reg = Word2Vec(size=n_dim, min_count=10)

        reddit_w2v_dep.build_vocab(x_train_dep)
        reddit_w2v_reg.build_vocab(x_train_reg)

        print('c. training model')
        #Train the model over train_reviews (this may take several minutes)
        reddit_w2v_dep.train(x_train_dep)
        reddit_w2v_reg.train(x_train_reg)

        reddit_w2v_dep.save('./dep_model.w2v')
        reddit_w2v_reg.save('./reg_model.w2v')

        reddit_w2v_dep = Word2Vec.load('./dep_model.w2v')
        reddit_w2v_reg = Word2Vec.load('./reg_model.w2v')

        print('d. scaling')
        print('scaling training')
        print('dep')
        train_vecs_p_dep = w2v_train_scale(n_dim, reddit_w2v_dep, x_train_dep)
        print('depn')
        train_vecs_n_dep = w2v_train_scale(n_dim, reddit_w2v_reg, x_train_dep)
        print('reg')
        train_vecs_p_reg = w2v_train_scale(n_dim, reddit_w2v_reg, x_train_reg)
        print('regn')
        train_vecs_n_reg = w2v_train_scale(n_dim, reddit_w2v_dep, x_train_reg)

        reddit_w2v_reg.train(x_test_reg)
        reddit_w2v_dep.train(x_test_dep)

        train_vecs_dep = np.concatenate((train_vecs_p_dep, train_vecs_n_dep), axis=1)
        train_vecs_reg = np.concatenate((train_vecs_n_reg, train_vecs_p_reg), axis=1)

        train_vecs = np.concatenate((train_vecs_dep, train_vecs_reg))

        print('scaling test vectors')
        #Build test tweet vectors then scale
        test_vecs_p_dep = np.concatenate([buildWordVector(z, n_dim, reddit_w2v_dep) for z in x_test_dep])
        test_vecs_n_dep = np.concatenate([buildWordVector(z, n_dim, reddit_w2v_reg) for z in x_test_dep])

        test_vecs_dep = np.concatenate((test_vecs_p_dep, test_vecs_n_dep), axis=1)
        #test_vecs_dep = scale(test_vecs_dep)

        test_vecs_p_reg = np.concatenate([buildWordVector(z, n_dim, reddit_w2v_reg) for z in x_test_reg])
        test_vecs_n_reg = np.concatenate([buildWordVector(z, n_dim, reddit_w2v_dep) for z in x_test_reg])

        test_vecs_reg = np.concatenate((test_vecs_n_reg, test_vecs_p_reg), axis=1)
        #test_vecs_reg = scale(test_vecs_reg)

        test_vecs = np.concatenate((test_vecs_dep, test_vecs_reg))

        print('e. logistical regression')
        #Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
        lr = run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('Simple NN')
        simpleNN(train_vecs, test_vecs, y_train, y_test, 0.01, 25, 100)

        print('f. plotting')
        show_graph(lr, test_vecs, y_test, split)






