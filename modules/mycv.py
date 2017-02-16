import numpy as np
from helper import load_image

def gradient(X):
    X = np.asarray(X, dtype = float)
    #X_padded = np.pad(X, ((2, 0), (2, 0)), 'reflect')
    #print X_padded
    #grad_X = X - X_padded[2:, :-2]
    #grad_Y = X - X_padded[:-2, 2:]
    grad_X = X[:, 1:] - X[:, :-1]
    grad_Y = X[1:, :] - X[:-1, :]
    return np.pad(grad_X, ((0, 0), (1, 0)), 'constant', constant_values=0), np.pad(grad_Y, ((1, 0), (0, 0)), 'constant', constant_values=0)

def card2polar(X, Y):
    mag = np.sqrt(X**2 + Y**2)
    ang = np.arctan2(Y, X)
    return mag, ang

def histogram(X):
    pass

def normalize(X):
    X = np.asarray(X, dtype = float)
    min_val = X.min()
    max_val = X.max()
    X = (X - min_val) / (max_val - min_val)
    return 255*X

def hog(X, pixels_per_cell = (8,8), block_norm = 'L1', transform_sqrt = True):
    # if X if a 1D vector
    if X.ndim == 1:
        n_dim = int(np.sqrt(len(X)))
        X = X.reshape((n_dim, n_dim))
    nx, ny = X.shape
    # normalize to [0,255]
    X = normalize(X)
    # take square-root of image
    if transform_sqrt:
        X = np.sqrt(X)
    # take gradient
    grad_X, grad_Y = gradient(X)
    # orientation of gradient
    _, ang = card2polar(grad_X, grad_Y)
    ang = (ang / np.pi + 1) * 180

    pcx, pcy = pixels_per_cell
    #TODO: handle cases where nx is not a multiple of pcx, etc
    ncx = int(np.floor(nx // pcx))
    ncy = int(np.floor(ny // pcy))

    hist = np.empty(0)
    for i in xrange(ncx):
        for j in xrange(ncy):
            ang_sub = ang[i*pcx:(i+1)*pcx, j*pcy:(j+1)*pcy].ravel()
            hst_sub, _ = np.histogram(ang_sub, bins = 8, range = (0, 360))
            hist = np.concatenate((hist, hst_sub))
    #hist /= np.sum(hist)
    return hist

if __name__ == '__main__':
    X = load_image('../data/Xtr.csv', max_rows = 2)
    X = X[0]
    hist = hog(X)
