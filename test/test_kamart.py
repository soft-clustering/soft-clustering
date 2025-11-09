import numpy as np
from soft_clustering._kmart._kmart import KMART


def test_kmart_basic():
    # Generate synthetic data
    N = 100
    D = 4
    X = np.vstack([
        np.random.uniform(0, 1, size=(N // 2, D)),
        np.random.uniform(0.5, 1.5, size=(N // 2, D))
    ])

    model = KMART(vigilance=0.75, learning_rate=0.5)
    labels, memberships = model.fit_predict(X)

    print("Cluster Labels:", labels)
    print("Membership Matrix (shape):", memberships.shape)


if __name__ == "__main__":
    test_kmart_basic()
