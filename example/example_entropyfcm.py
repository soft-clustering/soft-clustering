from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import ENTROPYFCM

    # Simulate simple 2D clusters
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.4, size=(40, 2))
    cluster2 = np.random.normal(loc=[3, 3], scale=0.4, size=(40, 2))
    X = np.vstack([cluster1, cluster2])

    model = ENTROPYFCM(c=2, m=2.0, entropy_weight=1.0, max_iter=100)
    labels, U = model.fit_predict(X)

    print("Labels:", labels)
    print("Membership matrix shape:", U.shape)
