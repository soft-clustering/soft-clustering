from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import AFCM

    # Generate synthetic data with 2 clusters
    N = 100
    D = 2
    X = np.vstack(
        [
            np.random.normal(loc=[0, 0], scale=0.5, size=(N // 2, D)),
            np.random.normal(loc=[4, 4], scale=0.5, size=(N // 2, D)),
        ]
    )

    model = AFCM(c=2, lambda_=1.0)
    labels, U = model.fit_predict(X)

    print("Consensus Cluster Labels:", labels)
    print("Membership Matrix U:", U)
