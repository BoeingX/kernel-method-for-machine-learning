#!/usr/bin/env python

"""KMeans.py: Perform k-means clustering."""

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
            for i in xrange(self.n_clusters):
                centers[i] = np.mean(X[labels == i], axis = 0)
            num_iter += 1

    def fit(self, X, y = None):
        min_loss = np.inf
        if self.compare_randomization:
            self.centers_all_ = np.empty((0, X.shape[1]))
            self.labels_all_ = np.empty((0))
            self.loss_all_ = np.empty((0))
        for i in xrange(self.n_init):
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

if __name__ == '__main__':

    try:
        data = np.loadtxt(sys.argv[1], delimiter = ' ')
    except:
        data = np.loadtxt('EMGaussian.data', delimiter = ' ')

    kmeans = KMeans(n_clusters = 4, compare_randomization = True)

    print kmeans.score(data)
    sys.exit(0)


    plt.figure()
    labels = kmeans.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c = labels, alpha = 0.3)
    centers = kmeans.centers_
    plt.scatter(centers[:, 0], centers[:, 1], c = range(len(centers)), s = 50)
    plt.tight_layout()
    plt.savefig('../report/imgs/q4a-1.pdf')
    
    plt.figure()
    centers_all = kmeans.centers_all_
    plt.scatter(data[:, 0], data[:, 1], c = 'gray', alpha = 0.3)
    idx_init = np.asarray([])
    for i in xrange(kmeans.n_init):
        idx_init = np.append(idx_init, np.ones(kmeans.n_clusters)*i)
    plt.scatter(centers_all[:, 0], centers_all[:, 1], c = idx_init, s = 60, alpha = 0.5)
    plt.tight_layout()
    plt.savefig('../report/imgs/q4a-2.pdf')

    loss_all = kmeans.loss_all_
    plt.figure()
    plt.plot(loss_all, 'o-')
    plt.xlabel('# of initialization')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig('../report/imgs/q4a-3.pdf')
