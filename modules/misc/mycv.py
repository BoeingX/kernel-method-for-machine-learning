import numpy as np
from matplotlib import pyplot as plt
from modules.misc.helper import bootstrap

try:
    xrange
except NameError:
    xrange = range

def gradient(X):
    """Calculate discrete gradient of an image by [-1, 0, 1] filter."""
    X = np.asarray(X, dtype = float)

    grad_X = np.empty_like(X)
    grad_X[:, 0] = 0
    grad_X[:, -1] = 0
    grad_X[:, 1:-1] = X[:, 2:] - X[:, :-2]

    grad_Y = np.empty_like(X)
    grad_X[0, :] = 0
    grad_X[-1, :] = 0
    grad_Y[1:-1, :] = X[2:, :] - X[:-2, :]
    return grad_X, grad_Y

def card2polar(X, Y):
    """Cartesian to polar coordinate system."""
    mag = np.sqrt(X**2 + Y**2)
    ang = np.rad2deg(np.arctan2(Y, X)) % 180
    return mag, ang

def normalize(X):
    """Normalize an image to [0,255]."""
    #TODO: RGB-image support
    X = np.asarray(X, dtype = float)
    min_val = X.min()
    max_val = X.max()
    X = (X - min_val) / (max_val - min_val)
    return 255 * X


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
        #for pixels_per_cell in [(5, 5), (10, 10), (30, 30)]:
        #    x_vec = np.concatenate((x_vec, transformer(img, pixels_per_cell = pixels_per_cell)))
        X_vec[i] = x_vec
    if y is not None:
        return X_vec, y
    else:
        return X_vec

def bow(X, transformer, y = None, bt = False):
    pass