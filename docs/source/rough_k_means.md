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
from rough_kmeans import RoughKMeans

class rough_kmeans.RoughKMeans(
    n_clusters: int = 2,
    weight_lower: float = 0.7,
    max_iter: int = 100,
    tol: float = 1e-4
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_rough_k_means.py#L5)

---

