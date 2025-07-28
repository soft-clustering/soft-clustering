import numpy as np
from soft_clustering._shbgf._shbgf import SHBGF


def test_shbgf_basic():
    # Simulate 3 soft clustering matrices for 100 samples and 3 clusters each
    N = 100
    soft1 = np.random.dirichlet(np.ones(3), size=N)
    soft2 = np.random.dirichlet(np.ones(3), size=N)
    soft3 = np.random.dirichlet(np.ones(3), size=N)

    model = SHBGF(n_clusters=3)
    labels = model.fit_predict([soft1, soft2, soft3])

    print("Consensus cluster labels:", labels)


if __name__ == "__main__":
    test_shbgf_basic()
