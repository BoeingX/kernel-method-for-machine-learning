import skimage
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
sys.path.append('modules')
from helper import load_image, load_label, train_test_split
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
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


if __name__ == '__main__':
    X = load_image('data/Xtr.csv')
    y = load_label('data/Ytr.csv')

    ids, lbs = BoW(X)
    tmp = DataFrame()
    tmp['lbs'] = lbs
    tmp = pd.get_dummies(tmp['lbs'])
    tmp['ids'] = ids
    tmp = tmp.groupby('ids').sum()
    X = tmp.as_matrix()

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    clf = SVC()
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)
