import numpy as np
import warnings
warnings.filterwarnings("error")

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

def hog(X, orientations = 9, pixels_per_cell = (8,8), transform_sqrt = False):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Parameters
    ----------
    X : (M, N) ndarray
        Input image (greyscale).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    transform_sqrt : bool, optional
        Apply power law compression to normalise the image before
        processing.

    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    """
    # If X if a 1D vector, converted to 2D form
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

    #TODO: use integral image to accelerate the histogram computation (c.f.: http://www.di.ens.fr/willow/teaching/recvis09/final_project/)
    cx, cy = pixels_per_cell
    ncx = int(np.floor(sx // cx))
    ncy = int(np.floor(sy // cy))

    hist = np.empty(0)
    for i in range(ncx):
        for j in range(ncy):
            ang_sub = ang[i*cx:(i+1)*cx, j*cy:(j+1)*cy].ravel()
            mag_sub = mag[i*cx:(i+1)*cx, j*cy:(j+1)*cy].ravel()
            hst_sub, _ = np.histogram(ang_sub, weights = mag_sub, bins = 9, range = (0, 180))
            # block normalization
            hst_sub /= (np.linalg.norm(hst_sub) + 1e-5)
            hist = np.concatenate((hist, hst_sub))
    return hist
