import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import PFCM

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.2, random_state=42)

# Fit the PFCM model
model = PFCM(n_clusters=3, random_state=42)
model.fit(X)

U = model.U
T = model.T
V = model.V
hard_labels = np.argmax(U, axis=0)
min_typicality = np.min(T, axis=0)

plt.figure(figsize=(10, 4))
plt.suptitle("PFCM Soft Clustering Test Example", fontsize=16)

plt.subplot(1, 2, 1)
plt.title("PFCM Clustering Result (Fuzzy Membership)")
plt.scatter(X[:, 0], X[:, 1], c=hard_labels, cmap='viridis', alpha=0.6)
plt.scatter(V[:, 0], V[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Typicality (Low = More Likely Outlier)")
plt.scatter(X[:, 0], X[:, 1], c=min_typicality, cmap='coolwarm', alpha=0.7)
plt.colorbar(label="Minimum Typicality")
plt.scatter(V[:, 0], V[:, 1], c='black', marker='X', s=200, label='Centroids')
plt.legend()

plt.tight_layout()
plt.show()