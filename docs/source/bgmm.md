# BGMM (Beta-Gaussian Mixture Model) Documentation

---
## 🔍 Overview
BGMM is a clustering algorithm that jointly models two types of data using Gaussian and Beta distributions. It is especially useful when integrating expression data (continuous, real-valued) and binding probability data (ranged in (0,1)).

---
## ⚙️ Class Definition

**Class Name:** `BGMM`

This class implements the EM algorithm to jointly fit Gaussian and Beta mixture components.

```python
class BGMM:
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-5):
        ...
```
---
## 📋 Parameters

| Parameter       | Type   | Default | Description                                   |
|----------------|--------|---------|-----------------------------------------------|
| `n_components` | int    | 3       | Number of clusters                            |
| `max_iter`     | int    | 100     | Maximum number of EM iterations               |
| `tol`          | float  | 1e-5    | Log-likelihood convergence threshold          |

---
## 🚀 Usage Examples

```python
from soft_clustering._bgmm._bgmm import BGMM
import numpy as np

# Generate synthetic data
Xg = np.concatenate([
    np.random.normal(0, 1, 50),
    np.random.normal(5, 1, 50)
])
Xb = np.concatenate([
    np.random.beta(2, 5, 50),
    np.random.beta(5, 2, 50)
])

model = BGMM(n_components=2, max_iter=100)
model.fit(Xg, Xb)

print("Labels:", model.predict())
print("Memberships:", model.predict_proba())
```

---

### 📥 Input / 📤 Output

- **Input to `fit(Xg, Xb)`**:
  - `Xg (np.ndarray)`: Gaussian data (shape: N)
  - `Xb (np.ndarray)`: Beta data (shape: N), values should be in (0,1)

- **Returns**:
  - Membership probabilities: `predict_proba()`
  - Cluster assignments: `predict()`

---

## 🛠️ Methods

- `fit(Xg, Xb)`: Runs the EM algorithm on mixed Gaussian + Beta data
- `predict_proba()`: Returns posterior membership probabilities (N x K)
- `predict()`: Returns hard cluster labels (N,)

---



## 📝 Implementation Notes

- Uses EM algorithm to jointly update Gaussian and Beta parameters
- Gaussian updates via standard weighted mean/variance
- Beta updates via method-of-moments estimation
- Numerically stable and interpretable for multi-modal biological data

---

### 📚 Reference

This implementation is based on:  
**"BGMM: A Beta-Gaussian Mixture Model for Clustering Genes with Multiple Data Sources"**  
by Liang et al.
