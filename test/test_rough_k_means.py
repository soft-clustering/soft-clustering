import numpy as np

if __name__ == '__main__':
    from soft_clustering import RoughKMeans
    # Generate two well-separated clusters
    X = np.vstack([
        np.random.randn(50, 2) + [0, 0],
        np.random.randn(50, 2) + [5, 5]
    ])
    model = RoughKMeans(n_clusters=2)
    result = model.fit(X)
    print('Converged in', result['n_iter'], 'iterations')
    print('Centroids:\n', result['centroids'])
