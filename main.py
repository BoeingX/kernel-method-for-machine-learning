import sys

import numpy as np

sys.path.append('modules')
from modules.classifier.svm import SVM
from modules.misc.mycv import img2vec
from modules.misc.input_output import load_image, load_label, save_label
from modules.misc.timefn import timefn
from modules.hog.hog import hog
#TODO: remove sklearn dependency
from sklearn.svm import SVC

try:
    xrange
except NameError:
    xrange = range

@timefn
def submission():
    print('[INFO] Loading data')
    X_train = load_image('data/Xtr.csv')
    X_test = load_image('data/Xte.csv')
    y_train = load_label('data/Ytr.csv')
    print('[INFO] Computing histogram of gradients')
    X_train_, y_train = img2vec(X_train, hog, y_train, bt = True, length = 144)
    X_test_ = img2vec(X_test, hog, length = 144)
    print('[INFO] Fitting SVM')
    clf = SVM(C = 2.15, gamma = 1.0, kernel = 'rbf')
    clf.fit(X_train_, y_train)
    print('[INFO] Predicting')
    y_test = clf.predict(X_test_)
    print('[INFO] Writing results to disk')
    save_label(y_test, 'data/Yte.csv')

def test(cv = 5):
    print('[INFO] Loading data')
    X = load_image('data/Xtr.csv')
    y = load_label('data/Ytr.csv')
    print '[INFO] Computing histogram of gradients'
    X_, y = img2vec(X, hog, y, bt = False, length = 144)

    from sklearn.model_selection import cross_val_score
    clf = SVM(C = 2.15, gamma = 1.0, kernel = 'rbf')
    scores = cross_val_score(clf, X_, y, cv=cv, n_jobs=-1)

    print(np.mean(scores))

def grid_search():
    from sklearn.model_selection import GridSearchCV
    X = load_image('data/Xtr.csv')
    y = load_label('data/Ytr.csv')
    print('[INFO] Computing histogram of gradients')
    X_, y = img2vec(X, hog, y, bt = False, length = 144)
    parameters = {'kernel': ['rbf'], 'C': np.linspace(0.01, 20, 20), 'gamma': np.linspace(1.0/200, 1, 20)}
    svc = SVC()
    clf = GridSearchCV(svc, parameters, n_jobs=-1, verbose=2)
    clf.fit(X_, y)
    print(clf.best_estimator_)
    print(clf.best_score_)

if __name__ == '__main__':
    test()
    #grid_search()
    #submission()
