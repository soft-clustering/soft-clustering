#  Possibilistic C-Means (PCM)

> **Robust soft clustering via typicalities, less sensitive to noise and outliers than standard FCM.**

---

## 🔍 Overview
**Possibilistic C-Means (PCM)** assigns each sample a **typicality** value per cluster (not forced to sum to 1).  
By decoupling cluster competition, PCM becomes more robust to outliers: atypical points get low typicalities across all clusters rather than being forced to belong somewhere.

---

## ⚙️ Class Definition

```python
class soft_clustering.PCM(
    random_state: int = None,
    m: float = 2.0,
    alpha: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    init: str = 'kmeans++'
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_pcm.py)

---

## 📋 Parameters

| Parameter       | Type   | Default   | Description                                                                                 |
|-----------------|--------|-----------|---------------------------------------------------------------------------------------------|
| `random_state`  | int    | None      | Seed for reproducible initialization and randomness.                                        |
| `m`             | float  | 2.0       | Fuzzifier (>1). Larger values yield softer typicalities.                                    |
| `alpha`         | float  | 1.0       | Scale factor in the update of per-cluster spread parameters `eta_k` (stability/robustness). |
| `max_iter`      | int    | 300       | Maximum number of update iterations.                                                        |
| `tol`           | float  | 1e-5      | Convergence tolerance on absolute improvement of the objective.                              |
| `init`          | str    | 'kmeans++'| Initialization strategy for centers: `'kmeans++'` or `'random'`.                            |

---

## 🚀 Usage Examples

```python
from soft_clustering import PCM
import numpy as np

np.random.seed(42)
n = 50
X1 = np.random.randn(n, 2) * 0.25 + np.array([0.0, 0.0])
X2 = np.random.randn(n, 2) * 0.25 + np.array([2.5, 2.5])
X = np.vstack([X1, X2])

K = 2  # number of clusters

model = PCM(random_state=42, max_iter=100, m=2.0, alpha=1.0, init='kmeans++')
typicalities = model.fit_predict(X, K)
print("Typicality matrix:\n", typicalities)
```

---

## 📥 Input / 📤 Output

- **Input to `fit_predict(X, K)`**
  - `X (np.ndarray or scipy.sparse)` of shape `(n_samples, n_features)`; sparse inputs are densified internally.
  - `K (int)`: number of clusters.

- **Returns**
  - `typicalities (np.ndarray)` of shape `(n_samples, K)`: possibilistic typicality degrees per sample (rows are **not** constrained to sum to 1).

---

## 🛠️ Methods

### `fit_predict(X, K)`

Fit the PCM model on the provided dataset and return the **typicality matrix**.  
PCM alternates between: (1) updating cluster centers using typicalities as weights, (2) updating per-cluster spreads `eta_k`, and (3) recomputing typicalities.

**Parameters:**
* `X` (`numpy.ndarray` or `scipy.sparse`, shape `(n_samples, n_features)`): Input data matrix. If sparse, it is converted to dense internally.
* `K` (`int`): Number of clusters.

**Returns:**
* `typicalities` (`np.ndarray`, shape `(n_samples, K)`): Possibilistic typicality degrees per sample (not row-normalized).

**Attributes set on the model:**
* `centers_` (`np.ndarray`, shape `(K, n_features)`): Final cluster centers.
* `typicalities_` (`np.ndarray`, shape `(n_samples, K)`): Final typicality matrix (same as return).
* `etas_` (`np.ndarray`, shape `(K,)`): Per-cluster spread parameters.
* `objective_trajectory_` (`np.ndarray`, shape `(t,)`): Objective values per iteration.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_pcm.py)

---

## 📝 Implementation Notes

- **Objective:** combines (i) typicality-weighted squared distances to centers and (ii) a penalty that discourages large typicalities away from centers via per-cluster scales `eta_k`.
- **Alternating updates:**
  - Update **centers** as typicality-weighted means (using `typicality^m`).
  - Update **etas** from the current typicalities and within-cluster distances (scaled by `alpha`).
  - Update **typicalities** using current centers and `eta_k`; typicalities are clipped to `(0, 1]`.
- **Not row-normalized:** typicalities do **not** have to sum to 1 across clusters (key difference from FCM).
- **Numerical stability:** small epsilon added to denominators and distances; typicalities are clipped to avoid zeros; `eta_k` lower-bounded.
- **Initialization:** centers via `'kmeans++'` (default) or `'random'`; initial `eta_k` from average within-cluster squared distances.
- **Convergence:** stop when absolute improvement in objective ≤ `tol`, or after `max_iter`.
- **Complexity (per iteration):** roughly `O(n * K * d)` time; memory `O(n * K + K * d)`.

---

## 📚 References

1. R. Krishnapuram, J. M. Keller (1993). **A Possibilistic Approach to Clustering.** *IEEE Transactions on Fuzzy Systems*, 1(2), 98–110.
