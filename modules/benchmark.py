from helper import load_image, load_label, train_test_split, img2vec
from svm import SVM
from skimage.feature import hog


if __name__ == '__main__':
    X = load_image('../data/Xtr.csv')
    y = load_label('../data/Ytr.csv')
    y -= 1
    X_ = img2vec(X, lambda x: hog(x, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1)))
    X_train, y_train, X_test, y_test = train_test_split(X_, y)
    clf = SVM()
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)
