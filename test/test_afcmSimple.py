import numpy as np
from soft_clustering._afcm_simple._afcm_simple import AFCMSimple


def test_afcm_simple_basic():
    # Generate synthetic data
    N = 100
    D = 2
    X = np.vstack([
        np.random.normal(loc=[0, 0], scale=0.5, size=(N // 2, D)),
        np.random.normal(loc=[3, 3], scale=0.5, size=(N // 2, D)),
    ])

    model = AFCMSimple(c=2)
    labels, U = model.fit_predict(X)

    print("Cluster labels:", labels)
    print("Membership matrix U:", U)


if __name__ == "__main__":
    test_afcm_simple_basic()
