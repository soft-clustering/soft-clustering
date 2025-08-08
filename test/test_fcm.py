# Overview of FCM usage
from os import path
import sys
import numpy as np


if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import FCM

    np.random.seed(42)
    n = 50
    X1 = np.random.randn(n, 2) * 0.2 + np.array([0.0, 0.0])
    X2 = np.random.randn(n, 2) * 0.2 + np.array([2.0, 2.0])
    X = np.vstack([X1, X2])

    K = 2  # number of clusters

    # Initialize and fit the model
    model = FCM(random_state=42, max_iter=50)

    memberships = model.fit_predict(X, K)
    print("Membership matrix:\n", memberships)
