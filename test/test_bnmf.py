import numpy as np
from soft_clustering._bnmf._bnmf import BayesianNMF


def test_bnmf():

    V = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0]
    ], dtype=float)

    model = BayesianNMF(n_clusters=2, max_iter=100)
    model.fit(V)
    W = model.get_membership()

    print("Soft Membership Matrix (W):")
    print(W)


if __name__ == "__main__":
    test_bnmf()
