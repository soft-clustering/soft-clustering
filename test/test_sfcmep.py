from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import SFCMEP

    # Generate synthetic data for 2 clusters
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=0, scale=0.5, size=(50, 2))
    X2 = rng.normal(loc=5, scale=0.5, size=(50, 2))
    X = np.vstack([X1, X2])

    # Semi-supervised labels (only a few given)
    y = np.array([0] * 5 + [None] * 45 + [1] * 5 + [None] * 45, dtype=object)

    model = SFCMEP(K=2, random_state=0, max_iter=50)
    result = model.fit_predict(X, y)

    U = result["membership_matrix"]
    V = result["centroids"]