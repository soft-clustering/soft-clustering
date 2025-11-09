# FCC (Fuzzy Color Clustering) Documentation

---

## 🔍 Overview
Fuzzy Color Clustering (FCC) is a color clustering algorithm that represents each cluster as a fuzzy color sphere in CIELAB space. Each data point (color) has a degree of membership in each cluster, based on its distance from the fuzzy centroid using a JND (Just Noticeable Difference) threshold.

---

## ⚙️ Class Definition
**Class Name:** `FCC`
This class implements the Fuzzy Color Clustering model.
```python
class FCC:
    def __init__(self, c: int, jnd: float = 20.0, max_iter: int = 100, tol: float = 1e-5):
        self.c = c
        self.jnd = jnd
        self.max_iter = max_iter
        self.tol = tol
```



---

## 📋 Parameters

| Parameter   | Type   | Default | Description                                              |
|-------------|--------|---------|----------------------------------------------------------|
| `c`         | int    | —       | Number of clusters                                       |
| `jnd`       | float  | 20.0    | Radius of fuzzy sphere (Just Noticeable Difference)      |
| `max_iter`  | int    | 100     | Maximum number of iterations                             |
| `tol`       | float  | 1e-5    | Convergence threshold for centroid updates               |

---
## 🚀 Usage Examples

```python
from soft_clustering._fcc._fcc import FCC
import numpy as np

# Generate synthetic CIELAB colors
X = np.vstack([
    np.random.normal(loc=[50, 20, 20], scale=5.0, size=(50, 3)),
    np.random.normal(loc=[70, -10, 30], scale=5.0, size=(50, 3))
])

model = FCC(c=2, jnd=20.0)
labels, U = model.fit_predict(X)

print("Labels:", labels)
print("Memberships:", U)
```

---

### 📥 Input / 📤 Output

- **Input to `fit_predict(X)`**:
  - `X (np.ndarray)`: Matrix of shape (N x 3) with color data in CIELAB space

- **Returns**:
  - `labels (np.ndarray)`: Final hard cluster assignments
  - `U (np.ndarray)`: Membership matrix with fuzzy degrees (N x C)

---

## 🛠️ Methods

- `__init__(...)`: Initializes the FCC model with number of clusters and fuzzy radius
- `fit_predict(X)`: Performs the clustering algorithm and returns both hard and soft results

---


## 📝 Implementation Notes

- Distance is calculated in CIELAB space (Euclidean)
- If a color is inside the fuzzy sphere → membership = 1
- Else → membership is computed with inverse of excess distance
- Memberships are normalized across clusters

---

### 📚 Reference

This implementation is based on:  
**"Fuzzy Color Model and Clustering Algorithm for Color Clustering"**  
by A. Kovács and J. Abonyi.
