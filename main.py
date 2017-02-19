import os
import sys
import sys
import numpy as np
from sklearn.datasets import make_classification
sys.path.append('modules')
from svm import SVM, binarySVM
from helper import binarize, pdist, preview, load_label, load_image, save_label, img2vec, train_test_split
from mycv import hog
from sklearn.svm import SVC

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
    X_train_ = img2vec(X_train, hog)
    X_test_ = img2vec(X_test, hog)
    print '[INFO] Fitting SVM'
    clf = SVM(C = 10)
    clf.fit(X_train_, y_train-1)
    print '[INFO] Predicting'
    y_test = clf.predict(X_test_)
    print '[INFO] Writing results to disk'
    save_label(y_test+1, 'data/Yte.csv')

def test():
    print '[INFO] Loading data'
    X = load_image('data/Xtr.csv')
    y = load_label('data/Ytr.csv')
    y -= 1
    print '[INFO] Computing histogram of gradients'
    X_ = img2vec(X, hog)
    X_train, y_train, X_test, y_test = train_test_split(X_, y)

    print '[INFO] Fitting SVM'
    clf = SVM(C = 10)
    clf.fit(X_train, y_train)
    print '[INFO] Predicting'
    print clf.score(X_train, y_train)


if __name__ == '__main__':
    #test()
    submission()
