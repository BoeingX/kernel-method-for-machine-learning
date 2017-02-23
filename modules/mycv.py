import numpy as np
import warnings
warnings.filterwarnings("error")

try:
    xrange
except NameError:
    xrange = range

def gradient(X):
    X = np.asarray(X, dtype = float)
    grad_X = np.zeros_like(X)
    grad_Y = np.zeros_like(X)
    grad_X[:, :-1] = X[:, 1:] - X[:, :-1]
    grad_Y[:-1, :] = X[1:, :] - X[:-1, :]
    #grad_X[:, :-2] = X[:, 2:] - X[:, :-2]
    #grad_Y[:-2, :] = X[2:, :] - X[:-2, :]
    return grad_X, grad_Y
    #return np.pad(grad_X, ((0, 0), (2, 0)), 'constant', constant_values=0), np.pad(grad_Y, ((2, 0), (0, 0)), 'constant', constant_values=0)

def card2polar(X, Y):
    mag = np.sqrt(X**2 + Y**2)
    ang = np.arctan2(Y, (X + 1e-15)) * (180 / np.pi) + 180
    ang[np.where(ang > 180)] -= 180
    return mag, ang

def normalize(X):
    X = np.asarray(X, dtype = float)
    min_val = X.min()
    max_val = X.max()
    X = (X - min_val) / (max_val - min_val)
    return 255*X

def hog(X, pixels_per_cell = (8,8), cells_per_block = (1, 1), orientations = 9, transform_sqrt = False):
    # if X if a 1D vector
    if X.ndim == 1:
        n_dim = int(np.sqrt(len(X)))
        X = X.reshape((n_dim, n_dim))
    sx, sy = X.shape
    # normalize to [0,255]
    X = normalize(X)
    # take square-root of image
    if transform_sqrt:
        X = np.sqrt(X)
    # take gradient
    grad_X, grad_Y = gradient(X)
    # orientation of gradient
    mag, ang = card2polar(grad_X, grad_Y)

    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    #TODO: handle cases where sx is not a multiple of cx, etc
    ncx = int(np.floor(sx // cx))
    ncy = int(np.floor(sy // cy))

    hist = np.empty(0)
    for i in range(ncx):
        for j in range(ncy):
            ang_sub = ang[i*cx:(i+1)*cx, j*cy:(j+1)*cy].ravel()
            mag_sub = mag[i*cx:(i+1)*cx, j*cy:(j+1)*cy].ravel()
            hst_sub, _ = np.histogram(ang_sub, weights = mag_sub, bins = 9, range = (0, 180))
            # block normalization
            hst_sub /= (np.sum(hst_sub) + 1e-5)
            hist = np.concatenate((hist, hst_sub))
    return hist
    #hist = np.zeros((ncx, ncy, orientations))
    #bins = [(180/orientations * i, 180/orientations * (i+1)) for i in range(orientations)]
    #from scipy.ndimage.filters import uniform_filter
    #for i in range(orientations):
    #    tmp = np.where((ang >= bins[i][0]) & (ang < bins[i][1]), ang, 0)
    #    cond = tmp > 0
    #    tmp = np.where(cond, mag, 0)
    #    hist[:, :, i] = uniform_filter(tmp, size=(cx, cy))[cx / 2::cx, cy / 2::cy].T
    #n_blocksx = (ncx - bx) + 1
    #n_blocksy = (ncy - by) + 1
    #hist_normalized = np.zeros((n_blocksx, n_blocksy,
    #                              bx, by, orientations))
    #for x in range(n_blocksx):
    #    for y in range(n_blocksy):
    #        block = hist[x:x + bx, y:y + by, :]
    #        eps = 1e-5
    #        hist_normalized[x, y, :] = block / np.sqrt(block.sum() ** 2 + eps)
    #return hist_normalized.ravel()


def bootstrap(X, y):
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
    ## reflection
    #X_ = np.zeros((sx*4, sy))
    #for i in range(sx):
    #    X_[4*i] = X[i]
    #    img = X[i].reshape(ndim, ndim)
    #    X_[4*i + 1] = np.fliplr(img).ravel()
    #    X_[4*i + 2] = np.flipud(img).ravel()
    #    X_[4*i + 3] = np.fliplr(np.flipud(img)).ravel()
    #return X_, np.repeat(y, 4)
