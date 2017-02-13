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
    svm = SVM(kernel = 'linear')
    svm.fit(X, y)
    print svm.predict(X)

    svc = SVC(kernel='linear')
    svc.fit(X, y)
    print svc.predict(X)
