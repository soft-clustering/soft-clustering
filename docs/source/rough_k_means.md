# Rough K-Means (RoughKMeans)

> **A clustering method where points can belong to multiple clusters, and centers are updated using both definite and possible members.**

---

## 🔍 Overview

`RoughKMeans` is a clustering algorithm based on rough-set theory. Instead of assigning each sample to exactly one cluster, it defines:

- **Lower approximation (L)**: samples that definitely belong to a cluster.
- **Upper approximation (U)**: samples that possibly belong to a cluster (including the lower set).

It computes per-cluster thresholds (`alpha` and `beta`) to distinguish core vs. fringe members, then updates cluster centroids by mixing core and fringe means.

---

## ⚙️ Class Definition

```python
class rough_kmeans.RoughKMeans(
    n_clusters: int = 2,
    weight_lower: float = 0.7,
    max_iter: int = 100,
    tol: float = 1e-4
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_rough_k_means.py#L5)

---

## 📋 Parameters

| Parameter    | Type   | Default | Description                                                               |
| ------------ | -------| ------- | ------------------------------------------------------------------------- |
| n_clusters   | `int`  | `2`     | Number of clusters to form.                                               |
| weight_lower | `float`| `0.7`   | Weight for averaging between lower and fringe regions when updating means.|
| max_iter     | `int`  | `100`   | Maximum number of iterations for convergence.                             |
| tol          | `float`| `1e-4`  | Tolerance for centroid movement to declare convergence.                   |

---

## 🚀 Usage Examples

```python
from soft_clustering import RoughKMeans
import numpy as np

# Create sample dataset
X = np.array([
    [1.0, 2.0],
    [1.2, 1.9],
    [0.8, 2.1],
    [8.0, 8.0],
    [8.2, 7.8],
    [7.9, 8.3]
])

# Initialize and fit the model
model = RoughKMeans(n_clusters=2, weight_lower=0.6, max_iter=50, tol=1e-3)
results = model.fit_predict(X)

print("Lower approximation:\n", results['lower_approx'])
print("Upper approximation:\n", results['upper_approx'])
print("Centroids:\n", results['centroids'])
print("Iterations:", results['n_iter'])
```

---

## 🛠️ Methods

### `fit_predict(X)`

Perform Rough K-Means clustering using interval-set approximations.

**Parameters:**

* `X` (`np.ndarray`, shape `(n_samples, n_features)`): Feature matrix of input data.

**Returns:**

* `result` (`dict`) with:
  
  `lower_approx`: (`np.ndarray`, shape `(n_samples, n_clusters)`) – Binary matrix indicating definite membership.
  
  `upper_approx`: (`np.ndarray`, shape `(n_samples, n_clusters)`) – Binary matrix indicating possible membership.
  
  `centroids`: (`np.ndarray`, shape `(n_clusters, n_features)`) – Final cluster centers.
  
  `n_iter`: (`int`) – Total number of iterations executed.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_rough_k_means.py#L14)

---

## 📝 Implementation Notes

* **Centroid Initialization:** Centroids are randomly initialized from the dataset.

---

## 📚 Reference

1. Lingras, P., & West, C. (2004). Interval Set Clustering of Web Users with Rough K-Means.
Journal of Intelligent Information Systems, 23(1), 5-16.(https://link.springer.com/article/10.1023/B:JIIS.0000029668.88665.1a)
