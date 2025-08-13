#  Gustafson–Kessel (GK)

> **Fuzzy clustering with adaptive full covariances—captures ellipsoidal clusters with varying shapes and orientations.**

---

## 🔍 Overview
**Gustafson–Kessel (GK)** extends fuzzy c-means by learning a **full covariance** for each cluster and measuring distances with a scaled Mahalanobis metric. This lets clusters stretch/rotate to fit anisotropic (elliptical) data while keeping a unit-determinant constraint on the induced metric.

---

## ⚙️ Class Definition

```python
class soft_clustering.GK(
    random_state: int = None,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    init: str = 'kmeans++',
    reg_covar: float = 1e-6
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_gk.py)

---

## 📋 Parameters

| Parameter       | Type   | Default    | Description                                                                                   |
|-----------------|--------|------------|-----------------------------------------------------------------------------------------------|
| `random_state`  | int    | None       | Seed for reproducible initialization and randomness.                                          |
| `m`             | float  | 2.0        | Fuzzifier (>1). Larger values yield softer memberships.                                       |
| `max_iter`      | int    | 300        | Maximum number of update iterations.                                                          |
| `tol`           | float  | 1e-5       | Convergence tolerance on absolute improvement of the objective.                                |
| `init`          | str    | 'kmeans++' | Initialization strategy for centers/memberships: `'kmeans++'` or `'random'`.                 |
| `reg_covar`     | float  | 1e-6       | Small ridge added to covariances for numerical stability (positive-definiteness).            |

---

## 🚀 Usage Examples

```python
from soft_clustering import GK
import numpy as np

np.random.seed(42)
n = 100

# Anisotropic clusters (elliptical) to exercise GK's adaptive covariance
A1 = np.array([[2.0, 0.5],
               [0.5, 0.3]])
A2 = np.array([[0.3, -0.4],
               [-0.4, 1.5]])

X1 = np.random.randn(n, 2) @ A1 + np.array([0.0, 0.0])
X2 = np.random.randn(n, 2) @ A2 + np.array([3.0, 3.0])
X = np.vstack([X1, X2])

K = 2  # number of clusters

model = GK(random_state=42, max_iter=100, m=2.0, init='kmeans++', reg_covar=1e-6)
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

Fit the GK model on the provided dataset and return the **membership matrix**.  
At each iteration GK:
1) estimates **fuzzy covariances** per cluster using memberships^m,  
2) computes **GK distances** with a scaled Mahalanobis metric,  
3) updates **memberships** and then **centers** (membership-weighted means).

**Parameters:**
* `X` (`numpy.ndarray` or `scipy.sparse`, shape `(n_samples, n_features)`): Input data matrix; if sparse, converted to dense internally.
* `K` (`int`): Number of clusters.

**Returns:**
* `memberships` (`np.ndarray`, shape `(n_samples, K)`): Row-normalized fuzzy memberships.

**Attributes set on the model:**
* `centers_` (`np.ndarray`, shape `(K, n_features)`): Final cluster centers.  
* `memberships_` (`np.ndarray`, shape `(n_samples, K)`): Final membership matrix (same as return).  
* `covariances_` (`np.ndarray`, shape `(K, n_features, n_features)`): Learned fuzzy covariances.  
* `metrics_A_` (`np.ndarray`, shape `(K, n_features, n_features)`): Cluster metrics with **unit determinant**.  
* `objective_trajectory_` (`np.ndarray`, shape `(t,)`): Objective values per iteration.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_gk.py)

---

## 📝 Implementation Notes

- **Distance metric (GK):** uses a **scaled Mahalanobis** distance defined by a per-cluster covariance `C_k` and a scale factor based on the covariance volume. Concretely, the squared distance is proportional to  
  `det(C_k)^(+1/d) * (x - c_k)^T C_k^{-1} (x - c_k)`,  
  which penalizes clusters with larger volumes appropriately and aligns with the standard GK formulation.
- **Unit-determinant metric:** the stored metric matrices `A_k` satisfy `det(A_k) = 1` by setting  
  `A_k = det(C_k)^(+1/d) * C_k^{-1}`.
- **Fuzzy covariance update:** covariances are computed as membership^m–weighted scatter matrices with a small ridge (`reg_covar`) added to the diagonal.
- **Numerical stability:** Cholesky factorization is used; if it fails, a tiny positive ridge is added before retrying; distances and denominators are protected with small epsilons.
- **Initialization:** `'kmeans++'` (default) starts from seeded centers and uniform memberships; `'random'` initializes memberships uniformly at random (row-normalized), then derives centers.
- **Convergence:** stop when the absolute improvement in the objective ≤ `tol`, or after `max_iter`.
- **Complexity (per iteration):** roughly `O(n * K * d^2)` time (due to full covariances) and `O(K * d^2 + n * K)` memory.

---

## 📚 References

1. D. E. Gustafson, W. C. Kessel (1979). **Fuzzy Clustering with a Fuzzy Covariance Matrix.** *IEEE CDC*.  
2. J. C. Bezdek (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*. Springer.
