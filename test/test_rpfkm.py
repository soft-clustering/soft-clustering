import numpy as np
from rpfkm import RPFKM  # Assuming RPFKM class is saved in rpfkm.py


def test_rpfkm_basic():
    from sklearn.datasets import make_blobs

    # Generate synthetic dataset
    X, _ = make_blobs(n_samples=100, centers=3, n_features=10, random_state=42)
    X = X.T  # Transpose to shape (D, N)

    # Initialize and run the RPFKM algorithm
    model = RPFKM(c=3, d=5, gamma=0.1, beta=1.0, max_iter=10)
    labels, U, W = model.fit_predict(X)

    print("Cluster labels:", labels)
    print("Membership matrix U shape:", U.shape)
    print("Projection matrix W shape:", W.shape)


if __name__ == "__main__":
    test_rpfkm_basic()
