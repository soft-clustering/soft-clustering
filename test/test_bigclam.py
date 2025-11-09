import numpy as np
from soft_clustering._bigclam._bigclam import BIGCLAM


def test_bigclam():

    adj = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0]
    ])

    model = BIGCLAM(n_nodes=6, n_communities=2,
                    max_iter=100, learning_rate=0.01)
    model.fit(adj)
    F = model.get_membership()

    print("Membership Matrix (F):")
    print(F)


if __name__ == "__main__":
    test_bigclam()
