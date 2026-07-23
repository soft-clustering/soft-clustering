from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import SoftKSC
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    y = np.where(y == 0, -1, 1)

    n_labeled = int(0.2 * len(X))
    X_labeled = X[:n_labeled]
    y_labeled = y[:n_labeled]
    X_unlabeled = X[n_labeled:]

    model = SoftKSC(gamma=2.0, C=1.0)
    model.fit(X_labeled, y_labeled, X_unlabeled)

    X_test = X
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    print("Predicted labels:")
    print(preds)

    print("\nSoft assignment probabilities:")
    print(probs)
