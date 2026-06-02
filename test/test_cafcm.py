from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import CAFCM

    np.random.seed(0)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[4, 4], scale=0.5, size=(50, 2))
    X = np.vstack([cluster1, cluster2])

    model = CAFCM(c=2, m_start=2.0, m_end=1.01, cooling_rate=0.95)
    labels, U = model.fit_predict(X)

    print("Labels:", labels)
    print("Memberships shape:", U.shape)
