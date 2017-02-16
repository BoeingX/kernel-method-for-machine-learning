import os
import sys
import sys
import numpy as np
from sklearn.datasets import make_classification
sys.path.append('modules')
from svm import SVM, binarySVM
from helper import binarize, pdist, preview, load_label, load_image, save_label
from mycv import hog

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

def img2vec(X, transformer):
    fds = np.empty((len(X), 128))
    for i, img in enumerate(X):
        fd = transformer(img)
        fds[i] = fd
    return fds

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

if __name__ == '__main__':
    submission()
