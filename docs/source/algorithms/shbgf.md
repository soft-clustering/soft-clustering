# SHBGF Algorithm Documentation

## 🔍 Overview
The Soft HBGF (sHBGF) is a consensus clustering algorithm that combines multiple soft clusterings by representing each data point with a concatenated membership vector and clustering them using KMeans. It is the soft version of the HBGF/hypergraph bipartite graph formulation.

---


## ⚙️ Class Definition
**Class Name:** `SHBGF`

```python
class SHBGF:
    def __init__(self, n_clusters: int, max_iter: int = 10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
```

Implements the soft HBGF algorithm that merges membership vectors from multiple soft clusterings and applies KMeans to produce consensus labels.

---

## 📋 Parameters
| Parameter     | Type  | Default | Description                                   |
|---------------|-------|---------|-----------------------------------------------|
| `n_clusters`  | int   | —       | Number of consensus clusters to output        |
| `max_iter`    | int   | 10      | Maximum number of iterations for KMeans       |

---

### 💻 Using Example

```python
from soft_clustering._shbgf._shbgf import SHBGF
import numpy as np

# Simulate 3 soft clusterings
soft1 = np.random.dirichlet(np.ones(3), size=100)
soft2 = np.random.dirichlet(np.ones(3), size=100)
soft3 = np.random.dirichlet(np.ones(3), size=100)

model = SHBGF(n_clusters=3)
labels = model.fit_predict([soft1, soft2, soft3])
print("Consensus Labels:", labels)
```

---

### 📥 Input / 📤 Output

- **Input to `fit_predict(soft_memberships)`**:
  - `soft_memberships (list of np.ndarray)`: List of soft clustering matrices (shape N x K)

- **Returns**:
  - `labels (np.ndarray)`: Consensus clustering labels (length N)

---

### 🔧 Methods

- `__init__(self, n_clusters: int, max_iter: int = 10)`: Initializes the model with number of clusters and KMeans iterations.
- `fit_predict(self, soft_memberships: List[np.ndarray])`: Concatenates membership vectors and applies KMeans to find consensus labels.

---



## 📝 Implementation Notes

- This method treats each data point as a concatenation of soft cluster memberships across different sources.
- The result is a single vector per object, which is clustered using KMeans.
- It avoids transforming soft memberships into hard labels, preserving more probabilistic information.

---

### 📚 Reference

1. Punera, K., & Ghosh, J. (2008). *Consensus-Based Ensembles of Soft Clusterings*.

