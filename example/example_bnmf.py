from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import BayesianNMF

    V = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=float,
    )

    model = BayesianNMF(n_clusters=2, max_iter=100)
    model.fit(V)
    W = model.get_membership()

    print("Soft Membership Matrix (W):")
    print(W)
