from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import AFCMSimple

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
