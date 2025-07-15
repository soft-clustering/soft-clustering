from os import path
import sys
import numpy as np


if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import SCM

    # Example 1: Separated 2D Clusters
    X1 = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [1.2, 2.2],
            [0.9, 1.7],
            [8.0, 8.0],
            [8.5, 8.2],
            [9.0, 7.8],
            [7.5, 8.1],
        ]
    )

    model1 = SCM(ra=1.0)
    centers1 = model1.fit(X1)

    print("Example 1: Separated 2D Clusters")
    print(f"Input data:\n{X1}")
    print(f"Detected cluster centers:\n{centers1}\n")

    # Example 2: Random 3D Clusters
    np.random.seed(42)
    cluster_a = np.random.normal(loc=[2, 2, 2], scale=0.3, size=(20, 3))
    cluster_b = np.random.normal(loc=[7, 7, 7], scale=0.3, size=(20, 3))
    X2 = np.vstack([cluster_a, cluster_b])

    model2 = SCM(ra=0.8)
    centers2 = model2.fit(X2)

    print("Example 2: Random 3D Clusters")
    print(f"Input data (first 5 points):\n{X2[:5]}")
    print(f"Detected cluster centers:\n{centers2}")
