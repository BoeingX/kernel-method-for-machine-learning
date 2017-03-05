import sys
sys.path.append('../modules/classifier')
from svm import SVM
from sklearn.datasets import make_classification
from sklearn.svm import SVC
if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_informative=5, n_classes = 10, random_state = 20)

    svm = SVM(kernel = 'linear', tol = 1e-10)
    svm.fit(X, y)
    print(svm.score(X, y))
    print(svm.bs)

    svc = SVC(kernel='linear')
    svc.fit(X, y)
    print(svc.score(X, y))
    print(svc.intercept_)
