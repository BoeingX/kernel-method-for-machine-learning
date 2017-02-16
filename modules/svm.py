import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_slsqp, minimize, fminbound
from base import Base
from helper import binarize, pdist, cdist

def _f(alpha, K, y):
    return -(2*np.dot(alpha, y) - np.dot(alpha, np.dot(K, alpha)))

def _f_grad(alpha, K, y):
    return -2*(y - np.dot(K, alpha))

def _find_b(alpha, K, y):
    b = 0
    f = lambda x: np.mean(np.maximum(np.zeros_like(y), np.ones_like(y) - np.multiply(y, x + np.dot(K, alpha))))
    b, _, _ = fmin_l_bfgs_b(f, b, approx_grad=True)
    return b

class binarySVM(Base):
    """Kernel SVM
    """
    def __init__(self, C = 1.0, gamma = 'auto', kernel = 'rbf', tol = 1e-5):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.tol = tol
        self._is_fitted = False


    def fit(self, X, y):
        self.X = X
        y = binarize(y)[:, 1]
        n_classes = len(np.unique(y))
        assert n_classes == 2
        n_samples, n_features = X.shape
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features
        if self.kernel == 'rbf':
            self.f = lambda x, y: np.exp(-np.sum(np.square(x-y))*self.gamma)
        elif self.kernel == 'linear':
            self.f = lambda x, y: np.dot(x, y)
        else:
            pass
        K = pdist(X, self.f)
        bounds = [(None, None)] * n_samples
        for j in xrange(n_samples):
            if y[j] == 1:
                bounds[j] = (0, self.C)
            else:
                bounds[j] = (-self.C, 0)
        alpha = np.random.rand(n_samples)
        cons = ({'type': 'eq',
                 'fun': lambda x: np.sum(x),
                 'jac': lambda x: np.ones_like(x)}
                )
        func = lambda x: _f(x, K, y)
        jac = lambda x: _f_grad(x, K, y)
        self.alphas = minimize(func, alpha, method = 'SLSQP', jac = jac, bounds = bounds, constraints=cons, tol = self.tol)['x']
        self.bs = _find_b(self.alphas, K, y)
        self._is_fitted = True

    def decision_function(self, X):
        K = cdist(self.X, X, self.f)
        y = np.sign(np.dot(self.alphas, K) + self.bs).squeeze()
        y = np.asarray(map(lambda x: x if x > 0 else 0, y), dtype = int)
        return y

    def predict(self, X):
        if not self._is_fitted:
            print '[Warning] Classifier is not yet fitted.'
        return self.decision_function(X)


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

    def fit(self, X, y):
        self.X = X
        n_classes = len(np.unique(y))
        n_samples, n_features = X.shape
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features
        if self.kernel == 'rbf':
            self.f = lambda x, y: np.exp(-np.sum(np.square(x-y))*self.gamma)
        elif self.kernel == 'linear':
            self.f = lambda x, y: np.dot(x, y)
        else:
            pass
        K = pdist(X, self.f)
        bounds = [(None, None)] * n_samples
        for j in xrange(n_samples):
            if y[j] == 1:
                bounds[j] = (0, self.C)
            else:
                bounds[j] = (-self.C, 0)
        y_bin = binarize(y)
        self.alphas = np.empty((n_classes, n_samples))
        self.bs = np.empty(n_classes)
        cons = ({'type': 'eq',
                 'fun': lambda x: np.sum(x),
                 'jac': lambda x: np.ones_like(x)}
                )
        for i in xrange(n_classes):
            print '[INFO] fitting class %d' % (i+1)
            yi = y_bin[:, i]
            func = lambda x: _f(x, K, yi)
            jac = lambda x: _f_grad(x, K, yi)
            bounds = [(None, None)] * n_samples
            for j in xrange(n_samples):
                if yi[j] == 1:
                    bounds[j] = (0, self.C)
                else:
                    bounds[j] = (-self.C, 0)
            alpha = np.zeros(n_samples)
            alpha = minimize(func, alpha, method = 'SLSQP', jac = jac, bounds = bounds, constraints=cons)
            self.alphas[i] = alpha['x']
            self.bs[i] = _find_b(alpha['x'], K, yi)
        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            print '[Warning] Classifier is not yet fitted.'
        K = cdist(self.X, X, self.f)
        fs = np.dot(self.alphas, K)
        fs += np.multiply(self.bs.reshape(-1, 1), np.ones_like(fs))
        y = np.argmax(fs, axis = 0)
        return y
