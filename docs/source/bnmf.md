# Bayesian NMF for Overlapping Community Detection


## 🔍 Overview
This algorithm performs overlapping community detection using Bayesian Non-negative Matrix Factorization (Bayesian NMF). It approximates a non-negative adjacency matrix V as a product of two lower-rank non-negative matrices W and H, and uses a Bayesian prior over the latent dimensions.

---


## ⚙️ Class Definition
**Class Name:** `BNMF`

This class implements a simple Bayesian NMF using multiplicative update rules and beta regularization.

```python
class BayesianNMF:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, a: float = 1.0, b: float = 1.0, tol: float = 1e-5):
        ...
```



---

## 📋 Parameters

| Parameter     | Type   | Default | Description                                          |
|----------------|--------|---------|------------------------------------------------------|
| `n_clusters`  | int    | 3       | Number of latent clusters (K)                        |
| `max_iter`    | int    | 100     | Maximum number of iterations                         |
| `a`           | float  | 1.0     | Gamma prior shape parameter                          |
| `b`           | float  | 1.0     | Gamma prior rate parameter                           |
| `tol`         | float  | 1e-5    | Convergence threshold for iterative updates          |

---
### 💻 Using Example

```python
from soft_clustering._bnmf._bnmf import BayesianNMF
import numpy as np

V = np.array([
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0]
], dtype=float)

model = BayesianNMF(n_clusters=2)
model.fit(V)
W = model.get_membership()

print(W)
```

---

### 📥 Input / 📤 Output

- **Input to `fit(V)`**:
  - `V (np.ndarray)`: Non-negative adjacency matrix (N x N)

- **Returns**:
  - Internal `W` and `H` matrices
  - Soft membership matrix via `get_membership()`

---

## 🛠️ Methods

- `fit(V)`: Performs Bayesian NMF on matrix `V`
- `get_membership()`: Returns the matrix `W` of node memberships (N x K)

---



## 📝 Implementation Notes

- Updates follow multiplicative rules
- Variational parameter `β` regularizes W and H
- Handles overlapping memberships naturally
- Fully unsupervised

---

### 📚 Reference

This implementation is based on:  
**"Overlapping Community Detection using Bayesian Non-negative Matrix Factorization"**  
by D. Yang, J. Liu, and X. Tang.
