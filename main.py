from __future__ import print_function
import numpy as np
from modules.classifier.svm import SVM
from modules.misc.mycv import img2vec
from modules.misc.input_output import load_image, load_label, save_label
from modules.misc.timefn import timefn
from modules.hog.hog import hog

@timefn
def submission():
    print('[INFO] Loading data')
    X_train = load_image('data/Xtr.csv')
    X_test = load_image('data/Xte.csv')
    y_train = load_label('data/Ytr.csv')
    print('[INFO] Computing histogram of gradients')
    X_train_, y_train = img2vec(X_train, hog, y_train, bt = True, length = 9*(1+4+16))
    X_test_ = img2vec(X_test, hog, length = 9*(1+4+16))
    print('[INFO] Fitting SVM')
    clf = SVM(C = 3.16, gamma = 1.0, kernel = 'rbf')
    clf.fit(X_train_, y_train)
    print('[INFO] Predicting')
    y_test = clf.predict(X_test_)
    print('[INFO] Writing results to disk')
    save_label(y_test, 'data/Yte.csv')

if __name__ == '__main__':
    submission()
