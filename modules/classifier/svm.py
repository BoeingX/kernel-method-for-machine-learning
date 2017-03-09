from __future__ import print_function
from itertools import combinations
import cvxopt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# Do not show optimization progress during optimization
cvxopt.solvers.options['show_progress'] = False
from cvxopt import matrix, solvers

from modules.classifier.base import Base
from modules.misc.helper import pdist, cdist

def _prepare_input_for_cvxopt(K, y, C, n):
    """Prepare input data for qp solver of cvxopt.

    qp solver takes the following problem as input:
    .. math::
        :nowrap:

        \begin{equation}
            \begin{aligned}
                \textrm{minimize}&\frac{1}{2}x'Px + q'x\\
                \textrm{subject to}& Gx\preceq h\\
                                    & Ax = b.
            \end{aligned}
        \end{equation}

    Parameters
    ----------
    K : (N, N) ndarray
        Kernel matrix.
    y : (N, ) ndarray
        Labels in {-1, 1}
    C : float
        Penalty parameter C of the error term.
    n : int
        Dimension of K (or y).
    Returns
    ----------
    P : (N, N) cvxopt matrix
    q : (N, ) cvxopt matrix
    G : (2N, N) cvxopt matrix
    h : (2N, ) cvxopt matrix
    A : (1, N) cvxopt matrix
    b : (1, ) cvxopt matrix
    """
    P = 2*matrix(K, tc = 'd')
    q = -2*matrix(y, tc = 'd')
    G1 = np.diag(y)
    G2 = np.diag(-y)
    G = matrix(np.concatenate((G1, G2), axis = 0), tc = 'd')
    h = matrix(np.concatenate((C*np.ones(n), np.zeros(n))), tc = 'd')
    A = matrix(np.ones((1, n)), tc = 'd')
    b = matrix(np.asarray([0]), tc = 'd')
    return P, q, G, h, A, b

def _f(alpha, K, y):
    """Dual problem of C-SVM."""
    return -(2*np.dot(alpha, y) - np.dot(alpha, np.dot(K, alpha)))

def _f_grad(alpha, K, y):
    """Gradient of dual C-SVM."""
    return -2*(y - np.dot(K, alpha))

def _find_b(alpha, K, y):
    """Find the optimal intercept given optimal alpha."""
    b = 0
    # Optimization problem of dual C-SVM with optimal alpha
    f = lambda x: np.mean(np.maximum(np.zeros_like(y), np.ones_like(y) - np.multiply(y, x + np.dot(K, alpha))))
    b, _, _ = fmin_l_bfgs_b(f, b, approx_grad=True)
    return b

class SVM(Base):
    """SVM classifier.
    """
    #TODO: sparse representation of alpha
    def __init__(self, C = 1.0, gamma = 'auto', kernel = 'rbf', tol = 0.001):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.tol = tol
        self._is_fitted = False

    def _fit_binary(self, X, y):
        """Solve binary SVM."""
        # Unique class labels
        y_unique = np.unique(y)
        # Sanity check: y should be in {-1, 1}
        assert np.array_equal(y_unique, np.asarray([-1, 1]))
        # Number of classes, should be 2 here.
        n_classes = len(y_unique)
        # Number of samples and features
        n_samples, n_features = X.shape
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features
        # Calculate kernel matrix
        K = pdist(X, self.kernel, self.gamma)
        # Prepare input for qp solver of cvxopt
        P, q, G, h, A, b  = _prepare_input_for_cvxopt(K, y, self.C, len(y))
        # Solve dual C-SVM
        sol = solvers.qp(P, q, G, h, A, b)
        # alpha
        alpha = np.asarray(sol['x']).ravel()
        # intercept
        b = _find_b(alpha, K, y)
        return alpha, b

    def fit(self, X, y):
        """Fit multi-class SVM using one-versus-one schema.

        The problem can be formulated as:
        .. math::
            :nowrap:
            \begin{equation}
                \begin{aligned}
                    \textrm{minimize}&\alpha'K\alpha - 2y'\alpha\\
                    \textrm{subject to}& \sum_{i=1}^n\alpha = 0\\
                                        & 0\leq y_i\alpha_i\leq C.
                \end{aligned}
            \end{equation}
        """
        # Register X and y since they are useful for prediction.
        self.X = X
        self.y = y
        y_unique = np.asarray(np.unique(y), dtype = int)
        n_classes = len(y_unique)
        n_samples, n_features = X.shape
        self.n_classes = n_classes
        # pairs are of the form [(0, 1), (0, 2), ..., (0, n-1), (1, 2), ..., (n-2, n-1)].
        self.pairs = list(combinations(y_unique, r = 2))
        # alphas and betas are dictionary storing alpha and b for each pair of class
        self.alphas = {}
        self.bs = {}
        #TODO: add multi-thread support
        # Loop over pairs
        for idx, pair in enumerate(self.pairs):
            print('[INFO] fitting class %d/%d' % (idx+1, len(self.pairs)))
            # Select the subset of training data whose label is in current pair
            #TODO: more efficient way to perform extraction
            idx1 = np.where(y == pair[0])
            idx2 = np.where(y == pair[1])
            X1 = X[idx1]
            y1 = np.ones(len(X1))
            X2 = X[idx2]
            y2 = -np.ones(len(X2))
            # Training data for binary SVM
            X_ = np.concatenate([X1, X2], axis = 0)
            y_ = np.concatenate([y1, y2])
            self.alphas[pair], self.bs[pair] = self._fit_binary(X_, y_)
        self._is_fitted = True

    def predict(self, X):
        """Predict labels for test data following one-versus-one schema."""
        if not self._is_fitted:
            print('[Warning] Classifier is not yet fitted.')
            #TODO: handle the error
        # Use majority vote for determining labels, count is used to store the votes.
        count = np.zeros((self.n_classes, len(X)))
        # Loop over pairs
        for pair in self.pairs:
            # Select the subset of training data whose label is in current pair
            #TODO: more efficient way to perform extraction
            idx1 = np.where(self.y == pair[0])
            idx2 = np.where(self.y == pair[1])
            X1 = self.X[idx1]
            X2 = self.X[idx2]
            X_ = np.concatenate([X1, X2], axis = 0)
            # Calculate the kernel matrix
            K = cdist(X_, X, self.kernel, self.gamma)
            # Evaluating :math:`f = \alpha K + b`.
            fs = np.dot(self.alphas[pair], K)
            # Add intercept
            fs += np.multiply(self.bs[pair], np.ones_like(fs))
            # Predicted label for the pair
            fs = np.sign(fs)
            # Update count
            #TODO: vectorize the operation.
            for idx, sgn in enumerate(fs):
                if sgn == 1:
                    count[pair[0], idx] += 1
                else:
                    count[pair[1], idx] += 1
        # Use majority vote for the final label.
        return np.argmax(count, axis = 0)
