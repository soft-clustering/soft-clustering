# GathGeva (Gath–Geva / Fuzzy Maximum-Likelihood Estimation)

## 🔍 Overview

Gath–Geva clustering replaces the Euclidean distance of fuzzy c-means with an
exponential distance derived from treating each cluster as a normal component
with its own covariance matrix and prior:

$$
d^2_{ik} \;=\; \frac{\sqrt{\det F_i}}{P_i}
    \exp\!\Big(\tfrac12 (x_k - v_i)^\top F_i^{-1} (x_k - v_i)\Big),
\qquad
P_i = \frac{1}{n}\sum_k u_{ik}^m .
$$

Because each cluster carries a full covariance, the method follows elongated
and differently-shaped clusters that fuzzy c-means — which assumes spherical
clusters of equal size — cuts across.

Memberships then follow the usual fuzzy c-means ratio rule applied to these
distances, so the rows of `memberships_` sum to one.

---

## ⚙️ Class Definition

```python
class GathGeva(BaseSoftClusterer):
    def __init__(self, n_clusters: int = 3, m: float = 2.0, max_iter: int = 100,
                 tol: float = 1e-5, reg_covar: float = 1e-6, init: str = "fcm",
                 init_iter: int = 20, random_state: int | None = None):
        ...
```

---

## 📋 Parameters

| Parameter      | Type          | Default  | Description |
|----------------|---------------|----------|-------------|
| `n_clusters`   | int           | 3        | Number of clusters |
| `m`            | float         | 2.0      | Fuzzifier, must be `> 1` |
| `max_iter`     | int           | 100      | Maximum alternating-optimisation sweeps |
| `tol`          | float         | 1e-5     | Convergence threshold on the largest membership change |
| `reg_covar`    | float         | 1e-6     | Relative ridge added to each fuzzy covariance |
| `init`         | `"fcm"` \| `"random"` | `"fcm"` | Membership initialisation |
| `init_iter`    | int           | 20       | Fuzzy c-means sweeps used by `init="fcm"` |
| `random_state` | int \| None   | None     | Seed for the initialisation |

---

## 🚀 Usage

```python
import numpy as np
from soft_clustering import GathGeva

rng = np.random.default_rng(0)
first = rng.normal(0, 1, (150, 2)) @ np.array([[3.0, 0.0], [0.0, 0.25]])
second = rng.normal(0, 1, (150, 2)) @ np.array([[0.25, 0.0], [0.0, 3.0]]) + 4.0
X = np.vstack([first, second])

model = GathGeva(n_clusters=2, random_state=0).fit(X)

U = model.memberships_            # (300, 2), rows sum to 1
centers = model.centers_          # (2, 2)
covariances = model.covariances_  # (2, 2, 2) fuzzy covariance per cluster
priors = model.priors_            # (2,) cluster priors, sum to 1
```

---

### 📥 Input / 📤 Output

- **Input to `fit(X)`**: `X (np.ndarray)` of shape `(n_samples, n_features)`.
- **Fitted attributes**: `memberships_`, `labels_`, `centers_`, `covariances_`,
  `priors_`, `n_iter_`, `n_clusters`.

---

## 🛠️ Methods

- `fit(X)` — run the algorithm and return `self`.
- `fit_predict(X)` — fit and return `memberships_`.
- `predict()` / `predict_proba()` — the fitted partition. Gath–Geva is
  transductive, so passing new data raises `NotImplementedError`.

---

## 📝 Implementation notes

**Why the initialisation matters.** The exponential distance has many poor
local optima, and Gath and Geva define the method as a *refinement* of a fuzzy
c-means partition rather than a standalone procedure. `init="fcm"` (the
default) runs `init_iter` fuzzy c-means sweeps first. On well-separated blobs
this is the difference between ARI 1.00 and ARI 0.24; `init="random"` exists
for ablation.

**Why it does not overflow.** Evaluated directly, the exponential term
overflows in double precision once the Mahalanobis distance exceeds roughly
1400, which happens routinely on outliers. The implementation works in log
space throughout,

$$
\log d^2_{ik} = \tfrac12 \log\det F_i - \log P_i
              + \tfrac12 (x_k - v_i)^\top F_i^{-1} (x_k - v_i),
$$

and computes memberships as
`softmax_i(-log d^2_ik / (m - 1))`, which is the same quantity without ever
forming the exponential. The log-determinant and the inverse both come from
one Cholesky factorisation; the ridge grows automatically if a cluster
collapses onto a lower-dimensional subspace.

---

### 📚 Reference

I. Gath and A. B. Geva. *Fuzzy clustering for the estimation of the parameters
of the components of mixtures of normal distributions.* Pattern Recognition
Letters, 9(2):77–86, 1989.
