#  Fuzzy C-Means (FCM)

> **A classic soft clustering algorithm assigning each sample fractional memberships across clusters.**

---

## 🔍 Overview
The **Fuzzy C-Means (FCM)** algorithm performs soft clustering by iteratively updating memberships and cluster centers. Unlike hard k-means, each sample gets fractional memberships across all clusters (rows sum to 1).

---

## ⚙️ Class Definition

```python
class soft_clustering.FCM(
    random_state: int = None,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    init: str = 'kmeans++'
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_fcm.py)

---

## 📋 Parameters

| Parameter       | Type   | Default   | Description                                                                  |
|-----------------|--------|-----------|------------------------------------------------------------------------------|
| `random_state`  | int    | None      | Seed for reproducible initialization and randomness.                         |
| `m`             | float  | 2.0       | Fuzzifier (>1). Larger values → softer memberships.                          |
| `max_iter`      | int    | 300       | Maximum number of update iterations.                                         |
| `tol`           | float  | 1e-5      | Convergence tolerance on absolute objective improvement.                     |
| `init`          | str    | 'kmeans++'| Initialization strategy: `'kmeans++'` or `'random'`.                         |

---

## 🚀 Usage Examples

```python
from soft_clustering import FCM
import numpy as np

np.random.seed(42)
n = 50
X1 = np.random.randn(n, 2) * 0.2 + np.array([0.0, 0.0])
X2 = np.random.randn(n, 2) * 0.2 + np.array([2.0, 2.0])
X = np.vstack([X1, X2])

K = 2  # number of clusters

# Initialize and fit the model
model = FCM(random_state=42, max_iter=50)

memberships = model.fit_predict(X, K)
print("Membership matrix:\n", memberships)
```

---

## 📥 Input / 📤 Output

- **Input to `fit_predict(X, K)`**
  - `X (np.ndarray or scipy.sparse)` of shape `(n_samples, n_features)`; sparse inputs are densified internally.
  - `K (int)`: number of clusters.

- **Returns**
  - `memberships (np.ndarray)` of shape `(n_samples, K)`: fuzzy membership degrees (each row sums to 1).

---

## 🛠️ Methods

### `fit_predict(X, K)`

Fit the FCM model on the provided dataset and return the fuzzy membership matrix.  
The algorithm alternates between updating memberships (row-normalized) and updating cluster centers as membership-weighted means until convergence or `max_iter` is reached.

**Parameters:**
* `X` (`numpy.ndarray` or `scipy.sparse`, shape `(n_samples, n_features)`): Input data matrix. If sparse, it is converted to dense internally.
* `K` (`int`): Number of clusters.

**Returns:**
* `memberships` (`np.ndarray`, shape `(n_samples, K)`): Fuzzy membership degrees per sample (rows sum to 1).

**Attributes set on the model:**
* `centers_` (`np.ndarray`, shape `(K, n_features)`): Final cluster centers.
* `memberships_` (`np.ndarray`, shape `(n_samples, K)`): Final membership matrix (same as return).
* `objective_trajectory_` (`np.ndarray`, shape `(t,)`): Objective values per iteration.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_fcm.py)

---

## 📝 Implementation Notes

- Minimizes the standard FCM objective: sum of (membership^m) times squared Euclidean distances to centers.
- Alternating optimization:
  - Update memberships from current centers; apply row-wise normalization so each row sums to 1.
  - Update centers as membership-weighted means (using memberships^m).
- Numerical stability: add a small epsilon to distances/denominators; clip memberships to be non-zero before normalization.
- Initialization: `'kmeans++'` (default) or `'random'` (random memberships normalized).
- Convergence: stop when the absolute improvement in objective ≤ `tol`, or after `max_iter`.
- Complexity (per iteration): roughly `O(n * K * d)` time and `O(n * K + K * d)` memory.

---

## 📚 References

1. J. C. Bezdek. *Pattern Recognition with Fuzzy Objective Function Algorithms*. Springer, 1981.  
2. J. C. Bezdek, R. Ehrlich, W. Full. “FCM: The Fuzzy c-Means Clustering Algorithm.” *Computers & Geosciences*, 10(2–3), 1984.
