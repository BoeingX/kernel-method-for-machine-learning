import sys
sys.path.append('../modules')
from sklearn.datasets import make_classification
from svm import SVM
from sklearn.svm import SVC
if __name__ == '__main__':
    X, y = make_classification(n_samples=5000, n_informative=5, n_classes = 2, random_state = 20)

    svm = SVM(kernel = 'rbf', tol = 1e-10)
    svm.fit(X, y)
    print svm.score(X, y)
    print svm.bs

    svc = SVC(kernel='rbf', decision_function_shape = 'ovr')
    svc.fit(X, y)
    print svc.score(X, y)
    print svc.intercept_
