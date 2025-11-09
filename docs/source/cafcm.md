#  CAFCM (Collaborative Annealing FCM) Documentation

---

## 🔍 Overview
CAFCM (Collaborative Annealing Fuzzy C-Means) is a fuzzy clustering algorithm that gradually sharpens membership assignments through an annealing process. It starts from a high fuzziness level and iteratively cools down the fuzziness parameter to encourage more deterministic (harder) cluster assignments.

---


## ⚙️ Class Definition
**Class Name:** `CAFCM`

```python
class CAFCM:
    def __init__(self, c: int, m_start: float = 2.0, m_end: float = 1.01,
                 cooling_rate: float = 0.95, max_iter: int = 100, tol: float = 1e-5):
        self.c = c
        self.m_start = m_start
        self.m_end = m_end
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.tol = tol
```

This class implements the collaborative annealing version of Fuzzy C-Means.

---

## 📋 Parameters
| Parameter       | Type   | Default | Description                                                   |
|------------------|--------|---------|---------------------------------------------------------------|
| `c`             | int    | —       | Number of clusters                                            |
| `m_start`       | float  | 2.0     | Initial fuzziness degree                                      |
| `m_end`         | float  | 1.01    | Final fuzziness degree (closer to 1 = more deterministic)     |
| `cooling_rate`  | float  | 0.95    | Multiplicative decay rate for `m` after each annealing step   |
| `max_iter`      | int    | 100     | Maximum iterations per fuzziness step                         |
| `tol`           | float  | 1e-5    | Convergence threshold for centroid updates                    |

---

### 💻 Using Example

```python
from soft_clustering._cafcm._cafcm import CAFCM
import numpy as np

X = np.vstack([
    np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[4, 4], scale=0.5, size=(50, 2)),
])

model = CAFCM(c=2, m_start=2.0, m_end=1.01, cooling_rate=0.95)
labels, U = model.fit_predict(X)

print("Labels:", labels)
print("Membership Matrix:", U)
```

---


### 📥 Input / 📤 Output

- **Input to `fit_predict(X)`**:
  - `X (np.ndarray)`: Input data array of shape (N x D)

- **Returns**:
  - `labels (np.ndarray)`: Final hard labels for each data point
  - `U (np.ndarray)`: Fuzzy membership matrix of shape (N x C)

---

## 🛠️ Methods

- `__init__(...)`: Initializes the model with cluster count and annealing settings
- `fit_predict(X)`: Runs the CAFCM algorithm and returns labels and membership matrix

---

## 📝 Implementation Notes

- Distance is calculated via Euclidean norm
- Memberships are updated based on current fuzziness value `m`
- Annealing process decreases `m` to push memberships toward binary
- Converges when centroids are stable under tolerance

---

### 📚 Reference

This implementation is based on:  
**"From Soft Clustering to Hard Clustering: A Collaborative Annealing Strategy"**
