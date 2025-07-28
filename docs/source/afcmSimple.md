# AFCM (Without Graph Embedding) Documentation

---

## 🔍 Overview
This version of Adaptive Fuzzy C-Means (AFCM) performs clustering by adaptively updating fuzzy memberships and cluster centers, without incorporating any graph structure. It is a simplified form of the full AFCM model and operates directly on the original data space.

---
## ⚙️ Class Definition
**Class Name:** `AFCMSIMPLE`

This class implements the simplified AFCM model with adaptive fuzzy memberships and convergence control.
```python
class AFCMSimple:
    def __init__(self, c: int, m: float = 2.0, max_iter: int = 100, tol: float = 1e-5):
        self.c = c
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
```

---

## 📋 Parameters

| Parameter     | Type   | Default | Description                                  |
|---------------|--------|---------|----------------------------------------------|
| `c`           | int    | —       | Number of clusters                           |
| `m`           | float  | 2.0     | Fuzziness degree                             |
| `max_iter`    | int    | 100     | Maximum number of iterations                 |
| `tol`         | float  | 1e-5    | Tolerance for convergence                    |

---
## 🚀 Usage Examples
```python
from soft_clustering._afcm_simple._afcm_simple import AFCMSimple
import numpy as np

X = np.vstack([
    np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[3, 3], scale=0.5, size=(50, 2)),
])

model = AFCMSimple(c=2)
labels, U = model.fit_predict(X)

print("Cluster Labels:", labels)
print("Membership Matrix:", U)
```

---

### 📥 Input / 📤 Output

- **Input to `fit_predict(X)`**:
  - `X (np.ndarray)`: Data matrix of shape (N x D)

- **Returns**:
  - `labels (np.ndarray)`: Final hard cluster assignments (N,)
  - `U (np.ndarray)`: Final fuzzy membership matrix (N x C)

---

## 🛠️ Methods

- `__init__(self, c, m=2.0, max_iter=100, tol=1e-5)`: Initializes the model with cluster count and training options.
- `fit_predict(self, X)`: Learns fuzzy memberships and returns both final labels and the U matrix.

---


## 📝 Implementation Notes

- This version does not include graph embedding or regularization terms.
- It iteratively updates fuzzy memberships and centers until convergence.
- Suitable for data without known manifold structure or graph connectivity.

---

## 📚 Reference

This implementation is based on:
**"Adaptive Fuzzy C-Means with Graph Embedding"**,  
but applies the simplified version without Laplacian constraints.
