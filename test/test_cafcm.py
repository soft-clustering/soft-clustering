import numpy as np
from soft_clustering._cafcm._cafcm import CAFCM


def test_cafcm_basic():

    np.random.seed(0)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[4, 4], scale=0.5, size=(50, 2))
    X = np.vstack([cluster1, cluster2])

    model = CAFCM(c=2, m_start=2.0, m_end=1.01, cooling_rate=0.95)
    labels, U = model.fit_predict(X)

    print("Labels:", labels)
    print("Memberships shape:", U.shape)


if __name__ == "__main__":
    test_cafcm_basic()
