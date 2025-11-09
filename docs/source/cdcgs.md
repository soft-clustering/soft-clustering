# CDCGS (Community Detection Clustering via Gumbel Softmax) Documentation

---

## 🔍 Overview
CDCGS is a graph clustering algorithm that applies the Gumbel-Softmax trick to learn soft assignments of nodes to communities. It computes a community relationship matrix R using these assignments and applies softmax to normalize interactions between clusters.

---

## ⚙️ Class Definition
**Class Name:** `CDCGS`
This class implements the Gumbel-Softmax based community clustering model.
```python
class CDCGS(nn.Module):
    def __init__(self, num_nodes: int, n_clusters: int, tau: float = 1.0):
        ...
```

---

## 📋 Parameters

| Parameter     | Type   | Default | Description                                     |
|----------------|--------|---------|-------------------------------------------------|
| `num_nodes`   | int    | —       | Number of graph nodes                           |
| `n_clusters`  | int    | —       | Number of target communities                    |
| `tau`         | float  | 1.0     | Gumbel-Softmax temperature                      |

---

## 🚀 Usage Examples

```python
from soft_clustering._cdcgs._cdcgs import CDCGS
import torch

adj = torch.tensor([
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0]
], dtype=torch.float)

model = CDCGS(num_nodes=6, n_clusters=2, tau=1.0)
R, soft_assign = model(adj)
loss = model.loss(R)

print("Loss:", loss.item())
```

---


## 🛠️ Methods

- `forward(adj)`: Computes community assignments and relation matrix
- `loss(output)`: Encourages the relation matrix to be close to identity

---



## 📝 Implementation Notes

- Learns soft cluster assignments via Gumbel-Softmax
- Encourages orthogonal (independent) clusters via identity loss
- Simple and fully unsupervised

---

### 📚 Reference

This implementation is based on:  
**"Community Detection Clustering via Gumbel Softmax"**  
by H. Zhang, J. Bu, Y. Wang, and C. Chen.
