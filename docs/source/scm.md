#  Subtractive Clustering Method (SCM)

> **A computationally efficient, density‑based algorithm for estimating cluster centers to initialize fuzzy inference systems.**

---

## 🔍 Overview

The **Subtractive Clustering Method (SCM)** algorithm estimates cluster centers by computing each point’s density‑based “potential” within a given radius, then iteratively selecting the highest‑potential points and suppressing their neighbors until a set of well‑spaced centers emerges—without needing to specify the number of clusters in advance.

---

## ⚙️ Class Definition

```python
class soft_clustering.NOCD(
    ra: float = 0.5
    ea: float = 0.5,
    er: float = 0.15
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_scm.py#L4)

---

## 📋 Parameters

| Parameter | Type    | Default | Description                                                               |
|-----------|---------|---------|---------------------------------------------------------------------------|
| `ra`      | `float` | `0.5`   | Neighborhood radius used to compute each point’s potential.               |
| `ea`      | `float` | `0.5`   | Acceptance ratio; terminates clustering when potential drops below this.  |
| `er`      | `float` | `0.15`  | Rejection ratio (reserved for extension; currently unused in core logic). |

---

## 🚀 Usage Examples

```python
from soft_clustering import SCM
import numpy as np

# Sample 2D data with two clusters
X = np.array([
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.2], [0.9, 1.7],
    [8.0, 8.0], [8.5, 8.2], [9.0, 7.8], [7.5, 8.1]
])

# Initialize and fit the model
model = SCM()
centers = model.fit(X)

print("Cluster centers found by SCM:")
print(centers)
```

---

## 🛠️ Methods

### `fit(X)`

Apply subtractive clustering to the input data and return cluster centers.

**Parameters:**

* `X` (`np.ndarray`, shape `(n_samples, n_features)`): Input data points.

**Returns:**

* `centers` (`np.ndarray`, shape `(n_clusters, n_features)`): Coordinates of the discovered cluster centers.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_scm.py#L23)

---

## 📚 Reference

1. Chiu, S. L. (1994). *Fuzzy Model Identification Based on Cluster Estimation*. In Proceedings of the IEEE International Conference on Fuzzy Systems. ACM Digital Library [2656640](https://dl.acm.org/doi/10.5555/2656634.2656640).
