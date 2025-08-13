# Overview of GK usage
from os import path
import sys
import numpy as np


if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import GK

    np.random.seed(42)
    n = 100

    # Anisotropic clusters (elliptical) to exercise GK's adaptive covariance
    A1 = np.array([[2.0, 0.5],
                   [0.5, 0.3]])
    A2 = np.array([[0.3, -0.4],
                   [-0.4, 1.5]])

    X1 = np.random.randn(n, 2) @ A1 + np.array([0.0, 0.0])
    X2 = np.random.randn(n, 2) @ A2 + np.array([3.0, 3.0])
    X = np.vstack([X1, X2])

    K = 2  # number of clusters

    # Initialize and fit the model
    model = GK(random_state=42, max_iter=100, m=2.0, init='kmeans++', reg_covar=1e-6)

    memberships = model.fit_predict(X, K)
    print("Membership matrix:\n", memberships)
