from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import SFCMEP

    X = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.15, 0.15],
        [0.9, 0.8],
        [0.8, 0.9],
        [0.85, 0.85],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])  # ground truth labels (semi-supervised)

    model = SFCMEP(K=2, random_state=42, max_iter=300)
    print(model.fit_predict(X, y))