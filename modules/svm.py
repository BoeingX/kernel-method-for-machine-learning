import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from cvxopt import matrix, solvers
from multiprocessing.dummy import Pool

from base import Base
from helper import binarize, pdist, cdist

def _prepare_input_for_cvxopt(K, y, C, n):
    P = 2*matrix(K, tc = 'd')
    q = -2*matrix(y, tc = 'd')
    G1 = np.diag(y)
    G2 = np.diag(-y)
    G = matrix(np.concatenate((G1, G2), axis = 0), tc = 'd')
    h = matrix(np.concatenate((C*np.ones(n), np.zeros(n))), tc = 'd')
    A = matrix(np.ones((1, n)), tc = 'd')
    b = matrix(np.asarray([0]), tc = 'd')
    return P, q, G, h, A, b

def _f(alpha, K, y):
    return -(2*np.dot(alpha, y) - np.dot(alpha, np.dot(K, alpha)))

def _f_grad(alpha, K, y):
    return -2*(y - np.dot(K, alpha))

def _find_b(alpha, K, y):
    b = 0
    f = lambda x: np.mean(np.maximum(np.zeros_like(y), np.ones_like(y) - np.multiply(y, x + np.dot(K, alpha))))
    b, _, _ = fmin_l_bfgs_b(f, b, approx_grad=True)
    return b

class SVM(Base):
    """Kernel SVM
    """
    #TODO: sparse representation of alpha
    def __init__(self, C = 1.0, gamma = 'auto', kernel = 'rbf', tol = 0.001):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.tol = tol
        self._is_fitted = False

    def _f(self, alpha, K, y):
        return -(2*np.dot(alpha, y) - np.dot(alpha, np.dot(K, alpha)))

    def _f_grad(self, alpha, K, y):
        return -2*(y - np.dot(K, alpha))

    def _find_b(self, alpha, K, y):
        b = 0
        f = lambda x: np.mean(np.maximum(np.zeros_like(y), np.ones_like(y) - np.multiply(y, x + np.dot(K, alpha))))
        b, _, _ = fmin_l_bfgs_b(f, b, approx_grad=True)
        return b

    def _fit_single(self, K, y_bin, i):
        print '[INFO] fitting class %d' % i
        y = y_bin[:, i]
        P, q, G, h, A, b  = _prepare_input_for_cvxopt(K, y, self.C, len(y))
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.asarray(sol['x']).ravel()
        b = _find_b(alpha, K, y)
        return np.concatenate((alpha, np.asarray(b)))

    def fit(self, X, y):
        self.X = X
        n_classes = len(np.unique(y))
        n_samples, n_features = X.shape
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features
#        if self.kernel == 'rbf':
#            self.f = lambda x, y: rbf(x, y, self.gamma)
#        elif self.kernel == 'linear':
#            self.f = linear
#        else:
#            pass
        K = pdist(X, self.kernel, self.gamma)
        y_bin = binarize(y)
        pool = Pool(4)
        alphas_bs = np.asarray(pool.map(lambda x: self._fit_single(K, y_bin, x), range(n_classes)))
        self.alphas = alphas_bs[:, :-1]
        self.bs = alphas_bs[:, -1]
        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            print '[Warning] Classifier is not yet fitted.'
        K = cdist(self.X, X, self.kernel, self.gamma)
        fs = np.dot(self.alphas, K)
        fs += np.multiply(self.bs.reshape(-1, 1), np.ones_like(fs))
        y = np.argmax(fs, axis = 0)
        return y
