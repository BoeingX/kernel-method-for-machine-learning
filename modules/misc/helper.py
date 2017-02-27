import numpy as np

try:
    xrange
except NameError:
    xrange = range

def binarize(y):
    if y.ndim != 1:
        print('[Warning] y is not a vector! Reshaping...')
        y = y.ravel()
    y = np.asarray(y, dtype = int)
    n_samples = len(y)
    n_classes = len(np.unique(y))
    y_bin = -np.ones((n_samples, n_classes))
    #TODO: handle the case where original labels are arbitrary
    for i in range(n_samples):
        y_bin[i][y[i]] = 1
    return y_bin

def pdist(X, kernel = 'rbf', gamma = 1.0, degree = 3, coef0 = 1.0):
    n = len(X)
    if kernel == 'rbf':
        A = np.multiply(np.square(np.linalg.norm(X, axis = 1)).reshape(-1, 1), np.ones((n, n)))
        B = np.dot(X, X.T)
        C = A.T
        return np.exp(gamma*(-A + 2*B - C))
    elif kernel == 'linear':
        return np.dot(X, X.T)
    elif kernel == 'polynomial':
        pass
    else:
        K = np.empty((n, n))
        if kernel == 'laplacian':
            f = lambda x, y: np.exp(-gamma*np.linalg.norm(x - y, ord = 1))
        elif kernel == 'intersection':
            f = lambda x, y: np.sum(np.minimum(x, y))
        else:
            pass
        for i in xrange(n):
            for j in xrange(i, n):
                d = f(X[i], X[j])
                K[i, j] = d
                K[j, i] = d
        return K

def cdist(X, Y, kernel = 'rbf', gamma = 1.0, degree = 3, coef0 = 1.0):
    n1 = len(X)
    n2 = len(Y)
    if kernel == 'rbf':
        A = np.multiply(np.square(np.linalg.norm(X, axis = 1)).reshape(-1, 1), np.ones((n1, n2)))
        B = np.dot(X, Y.T)
        C = np.multiply(np.square(np.linalg.norm(Y, axis = 1)).reshape(1, -1), np.ones((n1, n2)))
        return np.exp(gamma*(-A + 2*B - C))
    elif kernel == 'linear':
        return np.dot(X, Y.T)
    elif kernel == 'polynomial':
        pass
    else:
        K = np.empty((n1, n2))
        if kernel == 'laplacian':
            f = lambda x, y: np.exp(-gamma*np.linalg.norm(x - y, ord = 1))
        elif kernel == 'intersection':
            f = lambda x, y: np.sum(np.minimum(x, y))
        for i in xrange(n1):
            for j in xrange(n2):
                K[i, j] = f(X[i], Y[j])
        return K

def train_test_split(X, y, test_ratio = 0.1):
    n_test = np.int(len(X) * test_ratio)
    idx_test = set(np.random.choice(len(X), n_test, replace = False))
    idx_train = filter(lambda x: x not in idx_test, range(len(X)))
    idx_test = list(idx_test)
    idx_train = list(idx_train)
    return X[idx_train, :], y[idx_train], X[idx_test, :], y[idx_test]

def bootstrap(X, y):
    """Bootstrap training data by shifting image by 1 pixel in four directions."""
    sx, sy = X.shape
    ndim = int(np.sqrt(sy))
    # translation
    X_ = np.zeros((sx*5, sy))
    for i in range(sx):
        X_[5*i] = X[i]
        img = X[i].reshape(ndim, ndim)
        X_[5*i + 1] = np.pad(img[:, 1:], ((0, 0), (0, 1)), 'constant', constant_values=0).ravel()
        X_[5*i + 2] = np.pad(img[:, :-1], ((0, 0), (1, 0)), 'constant', constant_values=0).ravel()
        X_[5*i + 3] = np.pad(img[1:, :], ((0, 1), (0, 0)), 'constant', constant_values=0).ravel()
        X_[5*i + 4] = np.pad(img[:-1, :], ((1, 0), (0, 0)), 'constant', constant_values=0).ravel()
    return X_, np.asarray(np.repeat(y, 5), dtype = int)
