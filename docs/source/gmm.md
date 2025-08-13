#  Gaussian Mixture Models (GMM)

> **Expectation–Maximization (EM) for probabilistic soft clustering with Gaussian components.**

---

## 🔍 Overview
**Gaussian Mixture Models (GMM)** model data as a finite mixture of Gaussians. Using the **EM algorithm**, the method alternates between estimating posterior responsibilities of components for each sample (soft assignments) and updating mixture parameters (weights, means, covariances).

---

## ⚙️ Class Definition

```python
class soft_clustering.GMM(
    covariance_type: str = 'full',
    reg_covar: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-3,
    init_params: str = 'kmeans++',
    random_state: int = None
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_gmm.py)

---

## 📋 Parameters

| Parameter        | Type    | Default     | Description                                                                                  |
|------------------|---------|-------------|----------------------------------------------------------------------------------------------|
| `covariance_type`| str     | `'full'`    | Covariance structure: `'full'`, `'diag'`, or `'spherical'`.                                  |
| `reg_covar`      | float   | `1e-6`      | Small positive value added to variances/diagonals for numerical stability.                   |
| `max_iter`       | int     | `100`       | Maximum number of EM iterations.                                                             |
| `tol`            | float   | `1e-3`      | Convergence tolerance on the absolute improvement of log-likelihood.                         |
| `init_params`    | str     | `'kmeans++'`| Initialization for means: `'kmeans++'` or `'random'`.                                        |
| `random_state`   | int     | `None`      | Seed for reproducible initialization and sampling.                                           |

---

## 🚀 Usage Examples

```python
from soft_clustering import GMM
import numpy as np

np.random.seed(42)
n = 100
X1 = np.random.randn(n, 2) * np.array([0.6, 0.3]) + np.array([0.0, 0.0])
X2 = np.random.randn(n, 2) * np.array([0.5, 0.8]) + np.array([3.0, 3.0])
X = np.vstack([X1, X2])

K = 2  # number of mixture components

model = GMM(random_state=42, max_iter=100, covariance_type='full', init_params='kmeans++')
responsibilities = model.fit_predict(X, K)
print("Responsibilities matrix:\n", responsibilities)
```

---

## 📥 Input / 📤 Output

- **Input to `fit_predict(X, K)`**
  - `X (np.ndarray or scipy.sparse)` of shape `(n_samples, n_features)`; sparse inputs are densified internally.
  - `K (int)`: number of mixture components.

- **Returns**
  - `responsibilities (np.ndarray)` of shape `(n_samples, K)`: posterior probabilities (soft assignments) that each sample belongs to each component (rows sum to 1).

---

## 🛠️ Methods

### `fit_predict(X, K)`

Fit the GMM with the EM algorithm on the provided dataset and return the **posterior responsibilities**.

**Parameters:**
* `X` (`numpy.ndarray` or `scipy.sparse`, shape `(n_samples, n_features)`): Input data matrix. If sparse, it is converted to dense internally.
* `K` (`int`): Number of Gaussian components in the mixture.

**Returns:**
* `responsibilities` (`np.ndarray`, shape `(n_samples, K)`): Posterior probabilities per sample (row-normalized soft assignments).

**Attributes set on the model:**
* `weights_` (`np.ndarray`, shape `(K,)`): Mixture weights.
* `means_` (`np.ndarray`, shape `(K, n_features)`): Component means.
* `covariances_`:
  - `'full'`: `np.ndarray`, shape `(K, n_features, n_features)`
  - `'diag'`: `np.ndarray`, shape `(K, n_features)`
  - `'spherical'`: `np.ndarray`, shape `(K,)`
* `responsibilities_` (`np.ndarray`, shape `(n_samples, K)`): Final responsibilities (same as return).
* `lower_bound_` (`float`): Final log-likelihood value.
* `log_likelihood_trajectory_` (`np.ndarray`, shape `(t,)`): Log-likelihood values per EM iteration.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_gmm.py)

---

## 📝 Implementation Notes

- **EM routine:**
  - **E-step:** compute log-probabilities under each Gaussian, add log-weights, use a stable **log-sum-exp** normalization to obtain responsibilities.
  - **M-step:** update `weights_`, `means_`, and `covariances_` using current responsibilities.
- **Covariance options:**
  - `'full'`: full covariance with Cholesky-based log-density evaluation (stable and efficient).
  - `'diag'`: diagonal variances per feature.
  - `'spherical'`: single variance per component.
- **Regularization:** `reg_covar` added to variances/diagonals to avoid singular matrices and ensure positive-definiteness.
- **Initialization:** means via `'kmeans++'` (default) or `'random'`; initial weights uniform; initial covariances from the global data variance.
- **Convergence:** stop when absolute improvement of total log-likelihood ≤ `tol`, or after `max_iter`.
- **Complexity (per iteration):** roughly `O(n * K * d)` time (density evaluation dominates); memory `O(n * K + K * d^2)` for `'full'`.

---

## 📚 References

1. A. P. Dempster, N. M. Laird, and D. B. Rubin (1977). **Maximum Likelihood from Incomplete Data via the EM Algorithm.** *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.  
2. C. M. Bishop (2006). *Pattern Recognition and Machine Learning*. Springer. (Chapter on Mixture Models and EM)
