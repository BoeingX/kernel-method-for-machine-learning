import numpy as np
import sys
sys.path.append('../modules')
from helper import load_image, load_label, train_test_split
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier

def detect(X):
    ids = []
    descriptors = []
    for i, img in enumerate(X):
        img = img.reshape(32, 32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        sift = cv2.xfeatures2d.SIFT_create()
        try:
            kpts = sift.detect(img)
            dpts = sift.compute(img, kpts)[1]
            ids.extend(i*np.ones(len(kpts)))
            descriptors.extend(dpts)
        except:
            ids.append(i)
            descriptors.extend(np.zeros((1,128)))
    return np.asarray(ids, dtype = int), np.asarray(descriptors)

def BoW(X):
    ids, dpts = detect(X)
    clf = KMeans(n_clusters=50)
    clf.fit(dpts)
    lbs = clf.labels_
    return ids, lbs

def HOG(X):
    fds = np.empty((len(X), 128))
    for i, img in enumerate(X):
        img = img.reshape(32, 32)
        fd = hog(img, pixels_per_cell=(8,8), cells_per_block=(1,1), orientations=8)
        fds[i] = fd
    return fds

def grid_search(X, y):
    from sklearn.model_selection import GridSearchCV
    #parameters = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 50, 100], 'gamma': [1.0/128, 1.0/64, 1.0/16, 1.0/8, 1.0/4, 0.5, 1, 2]}
    parameters = {'kernel': ['rbf'], 'C': np.linspace(0.1, 10, 100)}
    svc = SVC()
    clf = GridSearchCV(svc, parameters, n_jobs=-1, cv = 3)
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    X = load_image('../data/Xtr.csv')
    y = load_label('../data/Ytr.csv')

    X_ = HOG(X)

    X_train, y_train, X_test, y_test = train_test_split(X_, y)

    clf = grid_search(X_, y)
    print clf.best_estimator_
    print clf.best_score_