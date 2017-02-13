import os
import sys
import sys
import numpy as np
from sklearn.datasets import make_classification
sys.path.append('modules')
from svm import SVM, binarySVM
from helper import binarize
from sklearn.svm import SVC

if __name__ == '__main__':
    X, y = make_classification(n_samples=100, n_informative=10, n_classes = 2, random_state=0)
    svm = binarySVM(kernel = 'rbf', tol = 1e-10, gamma = 0.1)
    svm.fit(X, y)
    a = np.sort(np.asarray(filter(lambda x: abs(x) > 1e-10, svm.alphas)))

    svc = SVC(kernel='rbf', gamma = 0.1)
    svc.fit(X, y)
    b = np.sort(svc.dual_coef_[0])
