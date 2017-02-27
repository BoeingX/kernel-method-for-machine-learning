import numpy as np
from modules.misc.mycv import normalize, gradient, card2polar

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