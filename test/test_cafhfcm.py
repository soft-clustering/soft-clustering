import numpy as np
from soft_clustering._cafhfcm._cafhfcm import CAFHFCM


def test_cafhfcm_basic():
    # Two simulated clusters in 2D space
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[1, 1], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    X = np.vstack([cluster1, cluster2])

    model = CAFHFCM(c=2, m=2.0, alpha=0.1)
    labels, memberships = model.fit_predict(X)

    print("Cluster Labels:", labels)
    print("Membership Matrix Shape:", memberships.shape)


if __name__ == "__main__":
    test_cafhfcm_basic()
