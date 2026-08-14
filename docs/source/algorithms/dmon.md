#  DMoN (Deep Modularity Networks) Documentation



## 🔍 Overview
DMoN is a deep clustering model for graphs that combines graph neural networks (GNNs) with modularity optimization. It performs soft clustering by maximizing modularity while preventing collapse using a regularization term. This approach enables unsupervised, end-to-end differentiable graph clustering.

---

## ⚙️ Class Definition
**Class Name:** `DMoN`
This PyTorch-based class implements the DMoN model using GCN layers and a soft assignment matrix.
```python
class DMoN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, n_clusters: int):
        ...
```

---

## 📋 Parameters

| Parameter      | Type   | Default | Description                                 |
|----------------|--------|---------|---------------------------------------------|
| `in_channels`  | int    | —       | Number of input features per node           |
| `hidden_channels` | int | —       | Number of hidden units in GCN               |
| `n_clusters`   | int    | —       | Desired number of clusters                  |

---
## 🚀 Usage Examples

```python
from soft_clustering._dmon._dmon import DMoN
import torch

x = torch.eye(6)
edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                           [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long)
adj = torch.zeros((6, 6))
for i, j in edge_index.t():
    adj[i, j] = 1

model = DMoN(in_channels=6, hidden_channels=8, n_clusters=2)
soft_assign = model(x, edge_index, adj)
loss = model.loss(soft_assign, adj)
```

---
### 📥 Input / 📤 Output

- **Input to `forward(x, edge_index, adj)`**:
  - `x (Tensor)`: Node feature matrix (N x F)
  - `edge_index (Tensor)`: Edge list for graph in COO format (2 x E)
  - `adj (Tensor)`: Dense adjacency matrix (N x N)

- **Returns**:
  - `soft_assign (Tensor)`: Cluster assignment probabilities (N x K)

---

## 🛠️ Methods

- `forward(x, edge_index, adj)`: Returns the soft cluster assignments.
- `loss(soft_assign, adj)`: Computes modularity loss and collapse regularization.

---



## 📝 Implementation Notes

- Uses GCNConv layers from `torch_geometric`
- Cluster assignments computed using softmax over final GCN output
- Modularity matrix is computed using degree-normalized formula
- Collapse regularization ensures diverse cluster use

---

### 📚 Reference

This implementation is based on:  
**"Graph Clustering with Graph Neural Networks"**  
by Alon, Yahav, and Wolf (2021).
