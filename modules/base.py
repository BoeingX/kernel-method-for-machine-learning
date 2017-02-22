import numpy as np


class Base():
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X, y):
        pass
    def score(self, X, y):
        y_pred = self.predict(X)
        assert len(y_pred) == len(y)
        return np.sum(y_pred == y) / np.float(len(y))
