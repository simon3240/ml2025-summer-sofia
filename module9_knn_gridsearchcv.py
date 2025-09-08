import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def read_dataset(name):
    n = int(input(f"{name} size: "))
    X = np.empty((n, 1), dtype=float)
    y = np.empty(n, dtype=int)
    for i in range(n):
        X[i,0] = float(input(f"{name} pair {i+1} - x: "))
        y[i]   = int(input(f"{name} pair {i+1} - y: "))
    return X, y


def main():
    X_train, y_train = read_dataset("Training")
    X_test,  y_test  = read_dataset("Test")

    Kmax = min(10, len(X_train))
    best_k, best_acc = 1, -1
    for k in range(1, Kmax+1):
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        if acc > best_acc:
            best_k, best_acc = k, acc

    print(f"Best k: {best_k}")
    print(f"Test accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
