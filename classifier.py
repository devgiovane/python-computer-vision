import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def bayes_train(x_train, y_train, x_test, y_test) -> None:
    [x_train, y_train] = [np.array(x_train), np.array(y_train)]
    [x_test, y_test] = [np.array(x_test), np.array(y_test)]
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    print_results("Naive", y_test, y_pred)


def knn_train(x_train, y_train, x_test, y_test) -> None:
    [x_train, y_train] = [np.array(x_train), np.array(y_train)]
    [x_test, y_test] = [np.array(x_test), np.array(y_test)]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print_results("KNN", y_test, y_pred)


def svm_train(x_train, y_train, x_test, y_test) -> None:
    [x_train, y_train] = [np.array(x_train), np.array(y_train)]
    [x_test, y_test] = [np.array(x_test), np.array(y_test)]
    svm = SVC(kernel='poly')
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    print_results("SVM", y_test, y_pred)


def print_results(name, y_test, y_pred):
    print(f"=== Metrics {name} ===")
    print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
    print(f"precision_score: {precision_score(y_test, y_pred, average='macro')}")
    print(f"recall_score: {recall_score(y_test, y_pred, average='macro')}")
    print(f"f1_score: {f1_score(y_test, y_pred, average='macro')}")
    print(f"=== Confusion {name} ===")
    print(confusion_matrix(y_test, y_pred))
