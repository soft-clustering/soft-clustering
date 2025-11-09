import numpy as np
from soft_clustering._fcc._fcc import FCC


def test_fcc_color_clustering():
    
    np.random.seed(0)
    cluster1 = np.random.normal(loc=[50, 20, 20], scale=5.0, size=(50, 3))
    cluster2 = np.random.normal(loc=[70, -10, 30], scale=5.0, size=(50, 3))
    X = np.vstack([cluster1, cluster2])

    model = FCC(c=2, jnd=20.0)
    labels, memberships = model.fit_predict(X)

    print("Labels:", labels)
    print("Membership matrix shape:", memberships.shape)


if __name__ == "__main__":
    test_fcc_color_clustering()
