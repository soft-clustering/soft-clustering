# SCSPA Algorithm Documentation

---

## 🔍Overview
The Soft CSPA (sCSPA) is a consensus clustering algorithm designed to combine multiple soft clustering results using similarity-based vector space modeling. It is the soft version of the classic CSPA algorithm.

---
## ⚙️ Class Definition
**Class Name:** `SCSPA`

Implements the soft CSPA algorithm that merges several soft membership matrices and applies KMeans in the combined space.


```python
class SCSPA:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
---
## 📋 Parameters

| Parameter     | Type  | Default | Description                              |
|---------------|-------|---------|------------------------------------------|
| `n_clusters`  | int   | —       | Number of consensus clusters to output   |

---
### 🚀Using Example

from soft_clustering._scspa._scspa import SCSPA
import numpy as np

# Simulate 3 soft clusterings
soft1 = np.random.dirichlet(np.ones(3), size=100)
soft2 = np.random.dirichlet(np.ones(3), size=100)
soft3 = np.random.dirichlet(np.ones(3), size=100)

model = SCSPA(n_clusters=3)
labels = model.fit_predict([soft1, soft2, soft3])
print("Consensus Labels:", labels)
### 📥 Input / 📤 Output

- **Input to `fit_predict(soft_memberships)`**:
  - `soft_memberships (list of np.ndarray)`: List of soft clustering matrices (shape N x K)

- **Returns**:
  - `labels (np.ndarray)`: Consensus clustering labels (length N)

---


### 🔧 Methods

- `__init__(self, n_clusters: int)`: Initializes the model with number of consensus clusters.
- `fit_predict(self, soft_memberships: List[np.ndarray])`: Concatenates soft membership matrices and applies KMeans to generate consensus clusters.
### 🛠️ Implementation Notes

- Each soft clustering is treated as a probabilistic embedding of objects.
- The vectors are concatenated and normalized before clustering.
- KMeans is used for deriving final cluster assignments.
- Cosine similarity is optionally computed for debugging purposes.

---

### 📚 Reference

This implementation is based on:
**"Consensus-Based Ensembles of Soft Clusterings" (sCSPA, sMCLA, sHBGF)**  
by Kunal Punera and Joydeep Ghosh.




