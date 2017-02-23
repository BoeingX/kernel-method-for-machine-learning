import numpy as np
import matplotlib.pyplot as plt
import time
from functools import wraps
from mycv import bootstrap

try:
    xrange
except NameError:
    xrange = range

def binarize(y):
    if y.ndim != 1:
        print '[Warning] y is not a vector! Reshaping...'
        y = y.ravel()
    y = np.asarray(y, dtype = int)
    n_samples = len(y)
    n_classes = len(np.unique(y))
    y_bin = -np.ones((n_samples, n_classes))
    #TODO: handle the case where original labels are arbitrary
    for i in xrange(n_samples):
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
        pass

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
        pass

def rgb2grayscale(R, G, B):
    return 0.2126*R + 0.7152*G + 0.0722*B

def preview(X, size = 9):
    """Preview grayscale image
    """
    # suppose size is perfect square
    assert np.sqrt(size).is_integer()
    width = np.int(np.sqrt(size))
    # TODO: fix imgs, choice only take 1d array
    idxs = np.random.choice(len(X), size = size, replace = False)
    for i, idx in enumerate(idxs):
        img = X[idx]
        if X[idx].ndim == 1:
            n_dim = np.int(np.sqrt(len(X[idx])))
            img = img.reshape(n_dim, n_dim)
        plt.subplot(width, width, i+1)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Data IO
def load_image(filename, grayscale = True, max_rows = None):
    if max_rows == 1:
        X = np.genfromtxt(filename, delimiter=',', max_rows = max_rows)[:-1]
    else:
        X = np.genfromtxt(filename, delimiter=',', max_rows = max_rows)[:, :-1]
    n_images, n_channels = X.shape
    n_dim = n_channels / 3
    if grayscale:
        R = X[:, :n_dim]
        G = X[:, n_dim:2*n_dim]
        B = X[:, 2*n_dim:]
        return rgb2grayscale(R, G, B)
    else:
        #TODO: handle RGB image
        pass

def load_label(filename):
    y = np.genfromtxt(filename, delimiter=',', skip_header=1)[:, 1]
    return y

def save_label(y, filename):
    n_samples = len(y)
    ids = np.asarray(range(1, n_samples + 1))
    Yte = np.empty((n_samples, 2))
    Yte[:, 0] = ids
    Yte[:, 1] = y
    with open(filename, 'w') as f:
        np.savetxt(f, Yte, '%d', delimiter = ',', newline = '\n', header = 'Id,Prediction', comments = '')

def train_test_split(X, y, test_ratio = 0.1):
    n_test = np.int(len(X) * test_ratio)
    idx_test = set(np.random.choice(len(X), n_test, replace = False))
    idx_train = filter(lambda x: x not in idx_test, xrange(len(X)))
    idx_test = list(idx_test)
    return X[idx_train, :], y[idx_train], X[idx_test, :], y[idx_test]

def img2vec(X, transformer, y = None, bt = False, length = 128):
    if bt and (y is not None):
        X, y = bootstrap(X, y)
    X_vec = np.empty((len(X), length))
    ndim = int(np.sqrt(len(X[0])))
    for i, img in enumerate(X):
        if img.ndim == 1:
            img = img.reshape(ndim, ndim)
        x_vec = transformer(img)
        #x_vec = np.empty(0)
        #for pixels_per_cell in [(8, 8), (16, 16), (32, 32)]:
        #    x_vec = np.concatenate((x_vec, transformer(img, pixels_per_cell = pixels_per_cell)))
        X_vec[i] = x_vec
    if y is not None:
        return X_vec, y
    else:
        return X_vec

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print ("@timefn:" + fn.func_name + " took " + str(t2 - t1) + " seconds")
        return result
    return measure_time
