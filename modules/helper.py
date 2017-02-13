import numpy as np
from multiprocessing.dummy import Pool
def binarize(y):
    if y.ndim != 1:
        print '[Warning] y is not a vector! Reshaping...'
        y = y.ravel()
    n_samples = len(y)
    n_classes = len(np.unique(y))
    print n_classes
    y_bin = -np.ones((n_samples, n_classes))
    #TODO: handle the case where original labels are arbitrary
    for i in xrange(n_samples):
        y_bin[i][y[i]] = 1
    return y_bin

def pdist(X, f):
    n_samples = len(X)
    K = np.empty((n_samples, n_samples))
    for i in xrange(n_samples):
        for j in xrange(i, n_samples):
            d = f(X[i], X[j])
            K[i, j] = d
            K[j, i] = d
    return K

def cdist(X, Y, f):
    n1 = len(X)
    n2 = len(Y)
    K = np.empty((n1, n2))
    for i in xrange(n1):
        for j in xrange(i, n2):
            d = f(X[i], Y[j])
            K[i, j] = d
            K[j, i] = d
    return K
