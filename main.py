import os
import sys
import sys
import numpy as np
from sklearn.datasets import make_classification
sys.path.append('modules')
from modules.svm import SVM
from modules.helper import binarize, pdist, preview, load_label, load_image, save_label, img2vec, train_test_split
from modules.timefn import timefn
from modules.mycv import hog
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
    #from skimage.feature import hog as skhog
    #from modules._hog import hog as skhog
    #X_, y = img2vec(X, lambda x: skhog(x, orientations=9, pixels_per_cell=(8,8), cells_per_block=(1,1), visualise=False), y, bt = False, length = 144)
    X_, y = img2vec(X, hog, y, bt = True, length = 144)

    from sklearn.model_selection import cross_val_score
    clf = SVC(C = 2.15, gamma = 1.0, kernel = 'rbf')
    scores = cross_val_score(clf, X_, y, cv=cv, n_jobs=-1)

    print(np.mean(scores))

    #scores_train = [None]*cv
    #scores = [None]*cv
    #for i in xrange(cv):
    #    print '[INFO] Fold No. %d' % (i+1)
    #    X_train, y_train, X_test, y_test = train_test_split(X_, y, 1.0 / cv)
    #    mean_img = np.mean(X_train, axis = 0)
    #    X_train -= mean_img
    #    X_test -= mean_img
    #    print '[INFO] Fitting SVM'
    #    clf = SVC(C = 10)
    #    clf.fit(X_train, y_train)
    #    print('[INFO] Predicting')
    #    scores_train[i] = clf.score(X_train, y_train)
    #    scores[i] = clf.score(X_test, y_test)
    #print(scores_train)
    #print(scores)
    #print(np.mean(scores))

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
    #test()
    #grid_search()
    submission()
