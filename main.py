import os
import sys
import sys
import numpy as np
from sklearn.datasets import make_classification
sys.path.append('modules')
from svm import SVM
from helper import binarize, pdist, preview, load_label, load_image, save_label, img2vec, train_test_split
from mycv import hog
from sklearn.svm import SVC
#from skimage.feature import hog

def test_svm():
    X, y = make_classification(n_samples=100, n_informative=5, n_classes = 2, random_state = 20)

    svm = SVM(kernel = 'linear', tol = 1e-10)
    svm.fit(X, y)
    print svm.score(X, y)
    print svm.bs

    svc = SVC(kernel='linear', decision_function_shape = 'ovr')
    svc.fit(X, y)
    print svc.score(X, y)
    print svc.intercept_

def test_io():
    X_train = load_image('data/Xtr.csv')
    X_test = load_image('data/Xte.csv')
    y_train = load_label('data/Ytr.csv')
    save_label(y_train, 'data/Yte.csv')

def submission():
    print '[INFO] Loading data'
    X_train = load_image('data/Xtr.csv')
    X_test = load_image('data/Xte.csv')
    y_train = load_label('data/Ytr.csv')
    print '[INFO] Computing histogram of gradients'
    X_train_, y_train = img2vec(X_train, hog, y_train, bt = True, length = 144)
    X_test_ = img2vec(X_test, hog, length = 144)
    print '[INFO] Fitting SVM'
    clf = SVM(C = 10)
    clf.fit(X_train_, y_train-1)
    print '[INFO] Predicting'
    y_test = clf.predict(X_test_)
    print '[INFO] Writing results to disk'
    save_label(y_test+1, 'data/Yte.csv')

def test(cv = 3):
    print '[INFO] Loading data'
    X = load_image('data/Xtr.csv')
    y = load_label('data/Ytr.csv')
    y -= 1
    print '[INFO] Computing histogram of gradients'
    X_, y = img2vec(X, hog, y, bt = True, length = 144)

    scores_train = [None]*cv
    scores = [None]*cv
    #for i in xrange(cv):
    for i in xrange(1):
        print '[INFO] Fold No. %d' % (i+1)
        X_train, y_train, X_test, y_test = train_test_split(X_, y, 1.0 / cv)
        print '[INFO] Fitting SVM'
        clf = SVC(C = 10)
        clf.fit(X_train, y_train)
        print '[INFO] Predicting'
        scores_train[i] = clf.score(X_train, y_train)
        scores[i] = clf.score(X_test, y_test)
    print scores_train
    print scores
    print np.mean(scores)

def grid_search():
    X = load_image('data/Xtr.csv')
    y = load_label('data/Ytr.csv')
    y -= 1
    print '[INFO] Computing histogram of gradients'
    #X_ = img2vec(X, lambda x: hog(x, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1)), y)
    X_, y = img2vec(X, hog, y, bt = True, length = 144)
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel': ['rbf'], 'C': np.linspace(0.1, 20, 20)}
    svc = SVC()
    clf = GridSearchCV(svc, parameters, n_jobs=-1, verbose=1)
    clf.fit(X_, y)
    print clf.best_estimator_
    print clf.best_score_

if __name__ == '__main__':
    #test()
    submission()
    #grid_search()
