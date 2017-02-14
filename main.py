import os
import sys
import sys
import numpy as np
from sklearn.datasets import make_classification
sys.path.append('modules')
from svm import SVM, binarySVM
from helper import binarize, pdist, preview, load_label, load_image
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
    #X_test = load_image('data/Xte.csv')
    #y_train = load_label('data/Ytr.csv')
    preview(X_train)
if __name__ == '__main__':
    test_io()
