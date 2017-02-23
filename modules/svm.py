from itertools import combinations
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import cvxopt
# Do not show optimization progress during optimization
cvxopt.solvers.options['show_progress'] = False
from cvxopt import matrix, solvers
from multiprocessing.dummy import Pool

from base import Base
from helper import binarize, pdist, cdist

try:
    xrange
except NameError:
    xrange = range

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

    def _fit_binary(self, X, y):
        y_unique = np.unique(y)
        assert np.array_equal(y_unique, np.asarray([-1, 1]))
        n_classes = len(y_unique)
        n_samples, n_features = X.shape
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features
        K = pdist(X, self.kernel, self.gamma)
        P, q, G, h, A, b  = _prepare_input_for_cvxopt(K, y, self.C, len(y))
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.asarray(sol['x']).ravel()
        b = _find_b(alpha, K, y)
        return alpha, b

    def fit(self, X, y):
        self.X = X
        self.y = y
        y_unique = np.unique(y)
        n_classes = len(y_unique)
        n_samples, n_features = X.shape
        self.n_classes = n_classes
        self.pairs = list(combinations(y_unique, r = 2))
        self.alphas = {}
        self.bs = {}
        # loop over pairs
        for pair in self.pairs:
            idx1 = np.where(y == pair[0])
            idx2 = np.where(y == pair[1])
            X1 = X[idx1]
            y1 = np.ones(len(X1))
            X2 = X[idx2]
            y2 = -np.ones(len(X2))
            X_ = np.concatenate([X1, X2], axis = 0)
            y_ = np.concatenate([y1, y2])
            self.alphas[pair], self.bs[pair] = self._fit_binary(X_, y_)
        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            print '[Warning] Classifier is not yet fitted.'
        count = np.zeros((self.n_classes, len(X)))
        for pair in self.pairs:
            idx1 = np.where(self.y == pair[0])
            idx2 = np.where(self.y == pair[1])
            X1 = self.X[idx1]
            X2 = self.X[idx2]
            X_ = np.concatenate([X1, X2], axis = 0)
            K = cdist(X_, X, self.kernel, self.gamma)
            fs = np.dot(self.alphas[pair], K)
            fs += np.multiply(self.bs[pair], np.ones_like(fs))
            fs = np.sign(fs)
            for idx, sgn in enumerate(fs):
                if sgn == 1:
                    count[pair[0], idx] += 1
                else:
                    count[pair[1], idx] += 1
        return np.argmax(count, axis = 0)
