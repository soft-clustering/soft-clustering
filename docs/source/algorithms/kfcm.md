#  Kernelized Fuzzy C-Means (KFCM)

> **A robust soft clustering algorithm using kernel-induced distances and intelligent centroid initialization.**

---

## 🔍 Overview

The **Kernelized Fuzzy C-Means (KFCM)** algorithm enhances standard fuzzy clustering by mapping data into a higher-dimensional feature space via a kernel function. This allows it to identify non-spherical cluster structures that traditional methods might miss.

This implementation is further improved with **K-Means++ initialization**, which ensures that the starting cluster centers are well-distributed, leading to faster convergence and more consistent, accurate results.

---

## ⚙️ Class Definition

```python
class soft_clustering.KFCM(
    n_clusters: int = 3,
    m: float = 2.0,
    sigma: float = 1.0,
    epsilon: float = 0.01,
    max_iter: int = 100
)
```

---

## 📋 Parameters

| Parameter      | Type            | Default | Description                                         |
| -------------- | --------------- | ------- | --------------------------------------------------- |
| `n_clusters`   | `int`           | `3`       | The number of clusters to form.                         |
| `m`            | `float`         | `2.0`   | The fuzziness exponent for the membership matrix. Must be > 1.                    |
| `sigma`          | `float`         | `1.0`   | The width (standard deviation) of the Gaussian kernel. A critical parameter that requires tuning.                    |
| `epsilon`            | `float`         | `0.01`   | The tolerance for convergence. The algorithm stops when the max change in memberships is below this value. |
| `max_iter`     | `int`           | `150`   | Maximum number of iterations.                       |

---

## 🚀 Usage Example

```python
import numpy as np
from soft_clustering import KFCM

# Create a sample dataset with three clusters
X = np.array([
    [1.1, 1.0], [1.5, 1.9], [0.9, 1.2],  # Cluster 1
    [6.0, 6.2], [6.5, 6.9], [5.9, 6.1],  # Cluster 2
    [9.5, 2.0], [9.1, 2.5], [8.9, 1.8]   # Cluster 3
])

# Initialize and train the model
# The 'sigma' value is tuned for this specific dataset.
model = KFCM(n_clusters=3, sigma=2.5)
labels = model.fit(X)

# Access results
print("Final Cluster Centers:\n", model.V)
print("\nPredicted Labels:", labels)
```
---

## 🛠️ Methods

### `fit(X)`
Trains the KFCM model on the input data `X`.

#### Parameters:

- `X` (`np.ndarray`): **Data matrix with shape `(n_samples, n_features)`**.

#### Returns:

- `labels` (`np.ndarray`): **An array of cluster labels for each data point**.

---

## 📝 Notes

- The performance of KFCM is highly sensitive to the `sigma` parameter. It should be tuned carefully based on the scale and distribution of your data.

- This implementation uses K-Means++ initialization to avoid poor convergence, making the results more reliable than a standard random start.

- Ideal for datasets where clusters may not be linearly separable in the original feature space.

---

## 📚 Reference

1. Dao-Qiang Zhang, Song-Can Chen (2004). *A novel kernelized fuzzy C-means algorithm with application in medical image segmentation*, 32, 37—50 (https://doi.org/10.1016/j.artmed.2004.01.012).