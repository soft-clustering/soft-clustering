from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import SoftDBSCANGM
    from sklearn.datasets import make_moons

    # Create data with complex shape (like moons)
    X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)

    model = SoftDBSCANGM(eps=0.3, min_samples=5, m=2.0, max_iter=100)
    model.fit(X)

    labels = model.predict()
    U = model.get_membership()

    print("Predicted Labels:")
    print(labels)

    print("\nFuzzy Membership Matrix:")
    print(U)
