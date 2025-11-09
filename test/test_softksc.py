import numpy as np
from sklearn.datasets import make_moons
from soft_clustering._soft_ksc._soft_ksc import SoftKSC


def test_soft_ksc():

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


if __name__ == "__main__":
    test_soft_ksc()
