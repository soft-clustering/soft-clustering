# CAF-HFCM (Centroid Auto-Fused Hierarchical FCM) Documentation

---

## 🔍 Overview
CAFHFCM is a fuzzy clustering algorithm that integrates traditional FCM with a centroid fusion regularization. The method promotes centroid merging by minimizing both distance-to-centroid loss and an L2 penalty encouraging centroid similarity.

---

## ⚙️ Class Definition
**Class Name:** `CAFHFCM`
This class implements the Centroid Auto-Fused Hierarchical FCM clustering algorithm.
```python
class CAFHFCM:
    def __init__(self, c: int, m: float = 2.0, alpha: float = 0.1,
                 max_iter: int = 100, tol: float = 1e-5):
        self.c = c
        self.m = m
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
```



---

## 📋 Parameters

| Parameter   | Type   | Default | Description                                             |
|-------------|--------|---------|---------------------------------------------------------|
| `c`         | int    | —       | Number of clusters                                      |
| `m`         | float  | 2.0     | Fuzziness coefficient                                   |
| `alpha`     | float  | 0.1     | Fusion regularization weight                            |
| `max_iter`  | int    | 100     | Maximum number of iterations                            |
| `tol`       | float  | 1e-5    | Tolerance for centroid convergence                      |

---

## 🚀 Usage Examples

```python
from soft_clustering._cafhfcm._cafhfcm import CAFHFCM
import numpy as np

X = np.vstack([
    np.random.normal(loc=[1, 1], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
])

model = CAFHFCM(c=2, m=2.0, alpha=0.1)
labels, memberships = model.fit_predict(X)

print("Labels:", labels)
print("Memberships:", memberships)
```

---

### 📥 Input / 📤 Output

- **Input to `fit_predict(X)`**:
  - `X (np.ndarray)`: Input data of shape (N x D)

- **Returns**:
  - `labels (np.ndarray)`: Hard cluster labels (N,)
  - `U (np.ndarray)`: Soft membership matrix (N x C)

---

## 🛠️ Methods

- `__init__(...)`: Initializes the model
- `fit_predict(X)`: Runs the CAFHFCM algorithm and returns labels and memberships

---


## 📝 Implementation Notes

- The algorithm updates membership matrix and centroids iteratively
- The centroid update includes a penalty for centroid differences
- Promotes centroid fusion when data distribution is hierarchical

---

### 📚 Reference

This implementation is based on:  
**"A Centroid Auto-Fused Hierarchical Fuzzy c-Means Clustering"**  
by L. Yu, Y. Pan, J. Wu, and J. Zhao.
