from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import FeMIFuzzy

    # Generate synthetic incomplete longitudinal data
    np.random.seed(0)
    X = np.random.rand(100, 5)
    mask = np.random.rand(*X.shape) < 0.1
    X[mask] = np.nan

    model = FeMIFuzzy(n_clusters=3)
    membership = model.fit_predict(X)

    print("Cluster membership matrix (U):")
    print(membership)

    print("Centroids:")
    print(model.centroids)
