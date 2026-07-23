from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import BIGCLAM

    adj = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]
    )

    model = BIGCLAM(n_nodes=6, n_communities=2, max_iter=100, learning_rate=0.01)
    model.fit(adj)
    F = model.get_membership()

    print("Membership Matrix (F):")
    print(F)
