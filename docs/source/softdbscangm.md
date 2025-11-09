# Soft DBSCAN-GM Documentation

---

## 🔍 Overview
Soft DBSCAN-GM is a fuzzy extension of the DBSCAN-GM algorithm. It combines density-based clustering with fuzzy logic by introducing membership degrees and iterative center updates using Mahalanobis distance.

---

## ⚙️ Class Definition
**Class Name:** `Soft DBSCAN-GM `
This class implements Soft DBSCAN-GM by first running DBSCAN and then refining membership degrees through fuzzy logic.
```python
class SoftDBSCANGM:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, m: float = 2.0,
                 max_iter: int = 100, tol: float = 1e-4):
        ...
```

---

## 📋 Parameters

| Parameter       | Type   | Default | Description                                                |
|----------------|--------|---------|------------------------------------------------------------|
| `eps`          | float  | 0.5     | DBSCAN epsilon radius for neighborhood detection           |
| `min_samples`  | int    | 5       | Minimum number of points for DBSCAN core points            |
| `m`            | float  | 2.0     | Fuzziness degree                                           |
| `max_iter`     | int    | 100     | Maximum number of fuzzy iterations                         |
| `tol`          | float  | 1e-4    | Tolerance for convergence of cluster centers               |

---
### 💻 Using Example

```python
from soft_clustering._soft_dbscan_gm._soft_dbscan_gm import SoftDBSCANGM
from sklearn.datasets import make_moons
import numpy as np

X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)

model = SoftDBSCANGM(eps=0.3, min_samples=5, m=2.0)
model.fit(X)

labels = model.predict()
membership = model.get_membership()

print("Labels:", labels)
print("Membership matrix:", membership)
```

---
### 📥 Input / 📤 Output

- **Input to `fit(X)`**:
  - `X (np.ndarray)`: Input data (N x D)

- **Returns**:
  - Fuzzy membership matrix: `get_membership()` → array (N x K)
  - Hard labels: `predict()` → array (N,)

---

 ## 🛠️ Methods

- `fit(X)`: Fits the clustering model to the data
- `get_membership()`: Returns the fuzzy membership matrix
- `predict()`: Returns hard labels for each sample

---



## 📝 Implementation Notes

- Initial clustering uses DBSCAN from scikit-learn
- Membership matrix initialized from DBSCAN labels
- Mahalanobis distance used in fuzzy updates
- Points marked as noise are treated as singleton clusters

---

### 📚 Reference

This implementation is based on:  
**"Fuzzy density-based clustering method: Soft DBSCAN-GM"**  
by X. Zhang, H. Xu, et al.
