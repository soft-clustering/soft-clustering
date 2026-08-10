# SFCMEP (Semi-supervised Fuzzy Clustering with Membership Prior)

## 🔍 Overview

SFCMEP is a semi-supervised fuzzy clustering algorithm. Alongside the feature
matrix it consumes a partially labelled target vector, converts those labels
into a *prior membership matrix*, and lets that prior steer the iterative fuzzy
partition. An expert-preference parameter `rho` controls how strongly the prior
pulls the solution towards the supplied labels, and an exponential distance
weighting governed by `lam` determines how quickly influence decays with
distance from a centroid.

Cluster `i` is initialised from the samples labelled `i`, so cluster indices
correspond to class indices — unlike a purely unsupervised method, the returned
partition is aligned with the label space you provided.

---

## ⚙️ Class Definition

**Class Name:** `SFCMEP`

```python
SFCMEP(
    K: int,
    random_state: int | None = None,
    max_iter: int = 200,
    rho: float = 0.5,
    lam: float = 1.0,
    tol: float = 1e-6,
)
```

---

## 📋 Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `K` | `int` | *required* | Number of clusters. Also accepted as `n_clusters`. |
| `random_state` | `int` or `None` | `None` | Seed used to initialise centroids for unlabelled classes. |
| `max_iter` | `int` | `200` | Maximum number of iterations. |
| `rho` | `float` | `0.5` | Expert preference: how strongly the prior membership constrains the partition. |
| `lam` | `float` | `1.0` | Scaling parameter of the exponential distance weighting. |
| `tol` | `float` | `1e-6` | Convergence tolerance on the change in memberships and centroids. |

---

## 📥 Input / 📤 Output

`fit_predict(X, y)` takes:

- `X` — `ndarray` of shape `(n_samples, n_features)`.
- `y` — label vector of length `n_samples`, with `None` for unlabelled samples.
  Use `dtype=object` so that `None` can sit alongside integer class labels.

It returns a dictionary with:

- `membership_matrix` — `ndarray` of shape `(n_samples, n_clusters)`.
- `centroids` — `ndarray` of shape `(n_clusters, n_features)`.

As with every SCPP estimator, the canonical attributes are populated after
fitting: `memberships_`, `labels_`, `centers_` and `n_clusters`.

---

## 🚀 Usage Examples

```python
import numpy as np
from soft_clustering import SFCMEP

rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal(loc=0, scale=0.5, size=(50, 2)),
    rng.normal(loc=5, scale=0.5, size=(50, 2)),
])

# Semi-supervised: only a handful of samples carry a label.
y = np.array([0] * 5 + [None] * 45 + [1] * 5 + [None] * 45, dtype=object)

model = SFCMEP(K=2, random_state=0, max_iter=50)
result = model.fit_predict(X, y)

U = result["membership_matrix"]   # (100, 2)
V = result["centroids"]           # (2, 2)

print("Membership matrix shape:", U.shape)
print("Cluster centers:\n", V)

# The canonical protocol attributes are available too.
print("Hard labels:", model.labels_[:10])
```

---

## 🛠️ Methods

### `fit_predict(X, y)`

Runs the semi-supervised optimisation and returns the membership matrix and
centroids. The prior membership matrix is built once from `y` and the initial
centroids, then memberships and centroids alternate until either `tol` or
`max_iter` is reached.

---

## 📝 Implementation Notes

- Samples labelled `None` contribute no prior and are clustered on the strength
  of the data alone; a fully `None` vector reduces the method to an
  unsupervised fuzzy partition with random centroid initialisation.
- If a class appears in `K` but no sample carries that label, its centroid falls
  back to a randomly chosen sample.
- The internal membership matrix is maintained as `(n_clusters, n_samples)` and
  transposed on return, so `membership_matrix` and `memberships_` are both
  `(n_samples, n_clusters)`.
