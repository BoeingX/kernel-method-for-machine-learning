from helper import load_image, load_label, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
import cv2

if __name__ == '__main__':
    X = load_image('../data/Xtr.csv')
    y = load_label('../data/Ytr.csv')
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    #clf = SVC(cache_size=1000)
    clf = KernelRidge(kernel = 'rbf')
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)
