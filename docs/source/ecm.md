#  ECM (Evidential C-Means) Documentation

---

## 🔍 Overview
ECM is an evidential extension of the classic Fuzzy C-Means algorithm. It is based on the theory of belief functions and produces a credal partition. Each data point can belong to multiple clusters or even be assigned to the ignorance/noise cluster if its assignment is uncertain.

---

## ⚙️ Class Definition
**Class Name:** `ECM`
This class implements the ECM clustering algorithm using mass assignment and credal partitions.

```python
class ECM:
    def __init__(self, n_clusters: int = 3, m: float = 2.0, delta: float = 10.0, max_iter: int = 100, tol: float = 1e-5):
        ...
```

---

## 📋 Parameters

| Parameter     | Type   | Default | Description                                                 |
|----------------|--------|---------|-------------------------------------------------------------|
| `n_clusters`  | int    | 3       | Number of clusters                                          |
| `m`           | float  | 2.0     | Fuzziness degree                                            |
| `delta`       | float  | 10.0    | Distance threshold for the noise cluster                   |
| `max_iter`    | int    | 100     | Maximum number of iterations                                |
| `tol`         | float  | 1e-5    | Convergence threshold based on centroid changes             |

---
## 🚀 Usage Examples

```python
from soft_clustering._ecm._ecm import ECM
import numpy as np

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
])

model = ECM(n_clusters=2, m=2.0, delta=5.0, max_iter=100)
model.fit(X)
mass = model.get_membership()

print("Mass matrix:")
print(mass)
```

---
### 📥 Input / 📤 Output

- **Input to `fit(X)`**:
  - `X (np.ndarray)`: Input data (N x D)

- **Returns**:
  - Mass matrix (N x K+1), where the last column is for the noise cluster

---

## 🛠️ Methods

- `fit(X)`: Runs the ECM algorithm and updates prototypes and mass matrix
- `get_membership()`: Returns the learned mass matrix including noise cluster

---



## 📝 Implementation Notes

- Mass values are normalized for each sample
- A noise cluster is modeled using a fixed distance `delta`
- Memberships are expressed as belief degrees, not just probabilities

---

### 📚 Reference

This implementation is based on:  
**"ECM: An Evidential Version of the Fuzzy C-Means Algorithm"**  
by T. Denoeux, M. Masson (2004).
