# SMCLA Algorithm Documentation

---

## 🔍 Overview
The Soft MCLA (sMCLA) is a consensus clustering algorithm that combines soft membership matrices by treating clusters as vectors and grouping them into meta-clusters. This is the soft version of the MCLA ensemble algorithm.

---


## ⚙️ Class Definition
**Class Name:** `SMCLA`
```python
class SMCLA:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
```

Implements the soft MCLA algorithm that aggregates memberships by clustering cluster-vectors themselves and re-assigning to objects.

---

## 📋 Parameters
| Parameter     | Type  | Default | Description                              |
|---------------|-------|---------|------------------------------------------|
| `n_clusters`  | int   | —       | Number of consensus clusters to output   |

---
## 🚀 Usage Examples

```python
from soft_clustering._smcla._smcla import SMCLA
import numpy as np

# Simulate 3 soft clusterings
soft1 = np.random.dirichlet(np.ones(3), size=100)
soft2 = np.random.dirichlet(np.ones(3), size=100)
soft3 = np.random.dirichlet(np.ones(3), size=100)

model = SMCLA(n_clusters=3)
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

- `__init__(self, n_clusters: int)`: Initializes the model with number of consensus clusters.
- `fit_predict(self, soft_memberships: List[np.ndarray])`: Transposes all cluster matrices, clusters the vectors, and aggregates to produce final labels.

--

### 🛠️ Implementation Notes

- Clusters are treated as feature vectors for meta-clustering.
- KMeans is applied to these vectors to create meta-clusters.
- Each data point is re-assigned based on aggregated contributions from cluster-vectors.
- Final assignment is the max-weighted cluster.

---

### 📚 Reference

This implementation is based on:
**"Consensus-Based Ensembles of Soft Clusterings" (sCSPA, sMCLA, sHBGF)**  
by Kunal Punera and Joydeep Ghosh.
