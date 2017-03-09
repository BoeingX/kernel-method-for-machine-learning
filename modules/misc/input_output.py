from __future__ import print_function
import numpy as np
from modules.misc.mycv import rgb2grayscale

def load_image(filename, grayscale = True, max_rows = None):
    if max_rows == 1:
        X = np.genfromtxt(filename, delimiter=',', max_rows = max_rows)[:-1]
    else:
        X = np.genfromtxt(filename, delimiter=',', max_rows = max_rows)[:, :-1]
    n_images, n_channels = X.shape
    n_dim = int(n_channels / 3)
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