import numpy as np

def linear(x, y):
    return np.dot(x, y)

def rbf(x, y, gamma = 1.0):
    return np.exp(-np.dot(x-y, x-y)*gamma)

def polynomial(x, y, degree = 3, gamma = 1.0, coef0 = 1.0):
    return np.power(gamma * np.dot(x, y) + coef0, degree)
