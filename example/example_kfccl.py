from os import path
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import KFCCL

    # Generate non-linear synthetic dataset
    X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)

    # Initialize KFCCL
    model = KFCCL(n_clusters=2, lambda_=10.0, gamma=5.0, epsilon=1e-4, max_iter=100)

    # Fit the model and get labels
    predicted_labels = model.fit(X)

    # Check output dimensions
    assert predicted_labels.shape[0] == X.shape[0], "Mismatch in number of predicted labels."

    print("Model output shapes:")
    print(f" - U shape: {model.U.shape}")
    print(f" - p_ik shape: {model.p_ik.shape}")
    print(f" - Kernel matrix shape: {model.K.shape}")

    # Calculate ARI (Adjusted Rand Index) for evaluation
    ari = adjusted_rand_score(y_true, predicted_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # Visualize clustering result
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='coolwarm', s=50, alpha=0.8)
    plt.title(f"KFCCL on Moons Dataset (ARI: {ari:.4f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

    print("KFCCL test on non-linear dataset completed.")
