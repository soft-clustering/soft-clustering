#  BIGCLAM (Cluster Affiliation Model for Big Networks) Documentation

---

## 🔍 Overview
BIGCLAM is an overlapping community detection algorithm based on non-negative matrix factorization. It models edge formation as a function of shared community affiliations and scales well to large networks.

---

## ⚙️ Class Definition
**Class Name:** `BIGCLAM`
This class implements the BIGCLAM model with coordinate gradient ascent and non-negative membership updates.

```python
class BIGCLAM:
    def __init__(self, n_nodes: int, n_communities: int, max_iter: int = 100, learning_rate: float = 0.01):
        ...
```

---

## 📋 Parameters

| Parameter         | Type   | Default | Description                                  |
|------------------|--------|---------|----------------------------------------------|
| `n_nodes`        | int    | —       | Number of nodes in the graph                 |
| `n_communities`  | int    | —       | Number of latent communities                 |
| `max_iter`       | int    | 100     | Maximum number of training iterations        |
| `learning_rate`  | float  | 0.01    | Learning rate for gradient updates           |

---
## 🚀 Usage Examples

```python
from soft_clustering._bigclam._bigclam import BIGCLAM
import numpy as np

adj = np.array([
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0]
])

model = BIGCLAM(n_nodes=6, n_communities=2)
model.fit(adj)
F = model.get_membership()
print(F)
```

---
### 📥 Input / 📤 Output

- **Input to `fit(adj)`**:
  - `adj (np.ndarray)`: Symmetric binary adjacency matrix (n x n)

- **Returns**:
  - Membership matrix `F` (n x k) via `get_membership()`

---

## 🛠️ Methods

- `fit(adj)`: Fits the BIGCLAM model to the input graph
- `get_membership()`: Returns the learned non-negative node-community matrix

---



## 📝 Implementation Notes

- Updates use block coordinate ascent
- Gradients computed with respect to edge and non-edge pairs
- Model enforces non-negativity of community affiliations
- Easily scales to large networks

---

### 📚 Reference

This implementation is based on:  
**"Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"**  
by J. Yang and J. Leskovec (2013).
