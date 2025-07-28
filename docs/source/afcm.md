# AFCM (Full Graph Embedding) Documentation

---

## 🔍 Overview
This version of Adaptive Fuzzy C-Means (AFCM) includes a graph embedding step using a Laplacian matrix derived from a k-nearest neighbors graph. The embedding regularizes fuzzy clustering using both manifold structure and adaptive memberships.

---


## ⚙️ Class Definition
**Class Name:** `AFCM`

```python
class AFCM:
    def __init__(self, c: int, lambda_: float = 1.0, m: float = 2.0, max_iter: int = 100, tol: float = 1e-5, n_neighbors: int = 5):
        self.c = c
        self.lambda_ = lambda_
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors = n_neighbors
```

This class implements the full AFCM algorithm with manifold embedding and adaptive fuzzy clustering.

---

## 📋 Parameters

| Parameter       | Type   | Default | Description                                          |
|------------------|--------|---------|------------------------------------------------------|
| `c`             | int    | —       | Number of clusters                                   |
| `lambda_`       | float  | 1.0     | Regularization strength for embedding                |
| `m`             | float  | 2.0     | Fuzziness degree                                     |
| `max_iter`      | int    | 100     | Maximum number of iterations                         |
| `tol`           | float  | 1e-5    | Tolerance for convergence                            |
| `n_neighbors`   | int    | 5       | Number of neighbors used to build Laplacian graph    |

---
## 🚀 Usage Examples

```python
from soft_clustering._afcm._afcm import AFCM
import numpy as np

X = np.vstack([
    np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[4, 4], scale=0.5, size=(50, 2)),
])

model = AFCM(c=2, lambda_=1.0)
labels, U = model.fit_predict(X)

print("Cluster Labels:", labels)
print("Membership Matrix:", U)
```

---

### 📥 Input / 📤 Output

- **Input to `fit_predict(X)`**:
  - `X (np.ndarray)`: Data matrix of shape (N x D)

- **Returns**:
  - `labels (np.ndarray)`: Final cluster assignments
  - `U (np.ndarray)`: Final fuzzy membership matrix

---

## 🛠️ Methods

- `__init__(...)`: Initializes the full AFCM model with graph embedding.
- `fit_predict(X)`: Applies fuzzy clustering in embedded space based on graph Laplacian and membership matrix.

---



## 📝 Implementation Notes

- The algorithm uses Laplacian + regularization term `B` to find a manifold-preserving embedding `X̃`.
- Embedding is updated at every iteration based on current membership matrix `U`.
- Suitable for datasets with complex or nonlinear structure.

---

### 📚 Reference

This implementation is based on:
**"Adaptive Fuzzy C-Means with Graph Embedding"**,  
Zhang, Liu, Liu, and Tao.
