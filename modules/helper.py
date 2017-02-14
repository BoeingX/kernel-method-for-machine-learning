import numpy as np
import matplotlib.pyplot as plt
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
        plt.subplot(width, width, i+1)
        n_dim = np.int(np.sqrt(len(X[idx])))
        plt.imshow(X[idx].reshape(n_dim, -1), cmap=plt.cm.binary)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Data IO
def load_image(filename, grayscale = True, max_rows = None):
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
