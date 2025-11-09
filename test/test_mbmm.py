import numpy as np
from soft_clustering._mbmm._mbmm import MBMM


def test_mbmm():
    np.random.seed(0)

    # ساخت داده‌های Beta برای 2 خوشه
    X1 = np.random.beta(2, 5, size=(50, 2))
    X2 = np.random.beta(5, 2, size=(50, 2))
    X = np.vstack([X1, X2])

    model = MBMM(n_components=2, max_iter=100)
    model.fit(X)

    labels = model.predict()
    probs = model.predict_proba()

    print("Predicted Labels:")
    print(labels)

    print("\nMembership Probabilities:")
    print(probs)


if __name__ == "__main__":
    test_mbmm()
