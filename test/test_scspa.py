from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import SCSPA

    # Simulate 3 soft clustering matrices for 100 samples and 3 clusters each
    N = 100
    soft1 = np.random.dirichlet(np.ones(3), size=N)
    soft2 = np.random.dirichlet(np.ones(3), size=N)
    soft3 = np.random.dirichlet(np.ones(3), size=N)

    model = SCSPA(n_clusters=3)
    labels = model.fit_predict([soft1, soft2, soft3])

    print("Consensus cluster labels:", labels)
