import os
import sys
import sys
import numpy as np
from sklearn.datasets import make_classification
sys.path.append('modules')
from svm import SVM, binarySVM
from helper import binarize, pdist
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel

if __name__ == '__main__':
    X, y = make_classification(n_samples=100, n_informative=2, n_classes = 2, random_state = 10)

    svm = binarySVM(kernel = 'linear', tol = 1e-10)
    svm.fit(X, y)
    print svm.score(X, y)
    print svm.bs

    svc = SVC(kernel='linear')
    svc.fit(X, y)
    print svc.score(X, y)
    print svc.intercept_
