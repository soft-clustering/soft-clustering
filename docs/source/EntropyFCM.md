# EntropyFCM (Entropy c-Means) Documentation

---

## 🔍 Overview
Entropy c-Means (EntropyFCM) is a fuzzy clustering algorithm designed to balance between compact clusters and fuzzy membership distributions. The objective function optimizes both compactness and entropy of membership values, allowing flexible cluster boundaries based on user-defined weighting.

---

## ⚙️ Class Definition
**Class Name:** `EntropyFCM`
This class implements the EntropyFCM
algorithm, supporting soft clustering with entropy regularization.

```python
class EntropyFCM:
    def __init__(self, c: int, m: float = 2.0, entropy_weight: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-5):
        self.c = c
        self.m = m
        self.entropy_weight = entropy_weight
        self.max_iter = max_iter
        self.tol = tol
```


---

## 📋 Parameters

| Parameter          | Type   | Default | Description                                                  |
|--------------------|--------|---------|--------------------------------------------------------------|
| `c`                | int    | —       | Number of clusters                                           |
| `m`                | float  | 2.0     | Fuzziness degree                                             |
| `entropy_weight`   | float  | 1.0     | Weight of the entropy regularization term                    |
| `max_iter`         | int    | 100     | Maximum number of iterations                                 |
| `tol`              | float  | 1e-5    | Convergence threshold for centroid changes                   |

---
## 🚀 Usage Examples

```python
from soft_clustering._EntropyFCM._EntropyFCM import EntropyFCM
import numpy as np

X = np.vstack([
    np.random.normal(loc=[0, 0], scale=0.4, size=(40, 2)),
    np.random.normal(loc=[3, 3], scale=0.4, size=(40, 2))
])

model = ECM(c=2, m=2.0, entropy_weight=1.0)
labels, memberships = model.fit_predict(X)

print("Labels:", labels)
print("Membership matrix:", memberships)
```

---
### 📥 Input / 📤 Output

- **Input to `fit_predict(X)`**:
  - `X (np.ndarray)`: Input data matrix of shape (N x D)

- **Returns**:
  - `labels (np.ndarray)`: Hard labels assigned to each point
  - `U (np.ndarray)`: Fuzzy membership matrix (N x C)

---

## 🛠️ Methods

- `__init__(...)`: Initializes the model and sets parameters
- `fit_predict(X)`: Runs the clustering algorithm and returns both labels and memberships

---



## 📝 Implementation Notes

- Memberships initialized using a Dirichlet distribution
- Objective balances compactness (distance to centroids) and entropy
- Entropy encourages fuzzy, overlapping memberships when needed
- Convergence based on stability of centroid positions

---

### 📚 Reference

This implementation is based on:  
**"Fuzzy Clustering to Identify Clusters at Different Levels of Fuzziness"**  
by Miguel A. Carreira-Perpiñán and Weiran Wang.
