import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

try:
    xrange
except NameError:
    xrange = range

class KMeans():
    def __kmeans_rand_init__(self, X, k):
        return X[np.random.choice(range(len(X)), k)]

    def __init__(self, n_clusters = 8, max_iter = 300, n_init = 10, tol = 1e-4, compare_randomization = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.compare_randomization = compare_randomization
        self.is_fitted_ = False

    def fit_(self, X, y = None):
        centers = self.__kmeans_rand_init__(X, self.n_clusters)
        num_iter = 1
        while True:
            dist = cdist(X, centers, 'sqeuclidean')
            labels = np.argmin(dist, axis = 1) 
            loss = np.sum(np.min(dist, axis = 1))
            if loss < self.tol or num_iter > self.max_iter:
                return centers, labels, loss
            for i in range(self.n_clusters):
                centers[i] = np.mean(X[labels == i], axis = 0)
            num_iter += 1

    def fit(self, X, y = None):
        min_loss = np.inf
        if self.compare_randomization:
            self.centers_all_ = np.empty((0, X.shape[1]))
            self.labels_all_ = np.empty((0))
            self.loss_all_ = np.empty((0))
        for i in range(self.n_init):
            centers, labels, loss = self.fit_(X)
            if self.compare_randomization:
                self.centers_all_ = np.append(self.centers_all_, centers, axis = 0)
                self.labels_all_ = np.append(self.labels_all_, labels)
                self.loss_all_ = np.append(self.loss_all_, loss)
            if loss < min_loss:
                self.centers_, self.labels_, self.loss_ = centers, labels, loss
        self.is_fitted_ = True

    def predict(self, X):
        if not self.is_fitted_:
            self.fit(X)
        return self.labels_ 

    def score(self, X, y = None):
        if not self.is_fitted_:
            self.fit(X)
        return self.loss_
