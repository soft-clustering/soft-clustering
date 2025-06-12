import numpy as np
from os import path
import sys

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import RoughKMeans
    # Generate two well-separated clusters
    X = np.vstack([
        np.random.randn(50, 2) + [0, 0],
        np.random.randn(50, 2) + [5, 5]
    ])
    model = RoughKMeans(n_clusters=2)
    result = model.fit_predict(X)
    print('Converged in', result['n_iter'], 'iterations')
    print('Centroids:\n', result['centroids'])
