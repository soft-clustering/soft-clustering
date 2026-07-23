from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import CAFHFCM

    # Two simulated clusters in 2D space
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[1, 1], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    X = np.vstack([cluster1, cluster2])

    model = CAFHFCM(c=2, m=2.0, alpha=0.1)
    labels, memberships = model.fit_predict(X)

    print("Cluster Labels:", labels)
    print("Membership Matrix Shape:", memberships.shape)
