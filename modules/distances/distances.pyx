# encoding: utf-8
# filename: distances.pyx
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: nonecheck=True, boundscheck=False, wraparound = False

import cython
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport exp

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef inline double linear(double[:] x, double[:] y, float gamma, int n) nogil:
    cdef double dot = 0.0
    for i in xrange(n):
        dot += x[i] * y[i]
    return dot

cdef inline double rbf(double[:] x, double[:] y, float gamma, int n) nogil:
    cdef double norm2 = 0.0
    for i in xrange(n):
        norm2 += (x[i] - y[i])**2
    return exp(-norm2)

cpdef double[:,:] pdist(double[:,:] X, float gamma):
    cdef int n_samples = len(X)
    cdef double[:,:] K = np.empty((n_samples, n_samples))
    cdef int i, j, n
    cdef double d
    n = len(X)
    with nogil:
        for i in prange(n_samples, schedule = 'dynamic'):
            for j in xrange(i, n_samples):
                d = rbf(X[i], X[j], gamma, n)
                K[i, j] = d
                K[j, i] = d
    return K

cpdef double[:,:] cdist(double[:,:] X, double[:,:] Y, float gamma):
    cdef:
        int n1, n2, i, j, n
        double[:,:] K
    n1 = len(X)
    n2 = len(Y)
    n = len(X[0])
    K = np.empty((n1, n2))
    with nogil:
        for i in prange(n1):
            for j in xrange(n2):
                K[i, j] = rbf(X[i], Y[j], gamma, n)
    return K
