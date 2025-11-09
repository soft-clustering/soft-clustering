import numpy as np
from soft_clustering._ecm._ecm import ECM


def test_ecm():
    # Sample 2D data (6 samples)
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ])

    model = ECM(n_clusters=2, m=2.0, delta=5.0, max_iter=100)
    model.fit(X)
    mass = model.get_membership()

    print("Mass Matrix (including noise cluster):")
    print(mass)


if __name__ == "__main__":
    test_ecm()
