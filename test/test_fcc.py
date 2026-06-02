from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import FCC

    np.random.seed(0)
    cluster1 = np.random.normal(loc=[50, 20, 20], scale=5.0, size=(50, 3))
    cluster2 = np.random.normal(loc=[70, -10, 30], scale=5.0, size=(50, 3))
    X = np.vstack([cluster1, cluster2])

    model = FCC(c=2, jnd=20.0)
    labels, memberships = model.fit_predict(X)

    print("Labels:", labels)
    print("Membership matrix shape:", memberships.shape)
