# encoding: utf-8
# filename: distances.pyx

import cython
import numpy as np
cimport numpy as np
from cython.parallel cimport nogil, prange

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef inline rbf(np.ndarray[DTYPE_t, ndim=1] X, np.ndarray[DTYPE_t, ndim=1] y, float gamma):
    return np.exp(-np.dot(x-y, x-y))

cpdef np.ndarray[DTYPE_t, ndim=2] pdist(np.ndarray[DTYPE_t, ndim=2] X, float gamma):
    cdef int n_samples = len(X)
    cdef np.ndarray[DTYPE_t, ndim=2] K = np.empty((n_samples, n_samples))
    cdef int i, j
    cdef double d
    with nogil:
        for i in prange(n_samples):
            for j in xrange(i, n_samples):
                d = rbf(X[i], X[j], gamma)
                K[i, j] = d
                K[j, i] = d
    return K

cpdef np.ndarray[DTYPE_t, ndim=2] cdist(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y, f):
    cdef:
        int n1, n2, i, j
        np.ndarray[DTYPE_t, ndim=2] K
    n1 = len(X)
    n2 = len(Y)
    K = np.empty((n1, n2))
    for i in xrange(n1):
        for j in xrange(n2):
            K[i, j] = f(X[i], Y[j])
    return K
