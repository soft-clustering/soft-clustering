#  Neural Overlapping Community Detection (NOCD)

> **A state-of-the-art Graph Convolutional Network for discovering overlapping communities.**

---

## 🔍 Overview

The **Neural Overlapping Community Detection (NOCD)** algorithm identifies overlapping communities in graph-structured data by learning node embeddings via a Graph Convolutional Network (GCN) and reconstructing the adjacency matrix with a Bernoulli decoder.

---

## ⚙️ Class Definition

```python
class soft_clustering.NOCD(
    random_state: int = None,
    hidden_sizes: List[int] = [128],
    weight_decay: float = 1e-2,
    dropout: float = 0.5,
    batch_norm: bool = True,
    lr: float = 1e-3,
    max_epochs: int = 500,
    balance_loss: bool = True,
    stochastic_loss: bool = True,
    batch_size: int = 20000
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/main/soft_clustering/_nocd.py)

---

## 📋 Parameters

| Parameter        | Type        | Default | Description                                               |
| ---------------- | ----------- | ------- | --------------------------------------------------------- |
| random\_state    | `int`       | `None`  | Seed for reproducible experiments.                        |
| hidden\_sizes    | `List[int]` | `[128]` | Sizes of hidden GCN layers for embedding complexity.      |
| weight\_decay    | `float`     | `1e-2`  | L2 regularization strength to prevent overfitting.        |
| dropout          | `float`     | `0.5`   | Dropout rate in GCN layers for robustness.                |
| batch\_norm      | `bool`      | `True`  | Enable batch normalization for stable learning.           |
| lr               | `float`     | `1e-3`  | Learning rate for the Adam optimizer.                     |
| max\_epochs      | `int`       | `500`   | Maximum training epochs (early stopping by default).      |
| balance\_loss    | `bool`      | `True`  | Balance contributions of edges and non-edges in the loss. |
| stochastic\_loss | `bool`      | `True`  | Use mini-batch (stochastic) or full-batch training.       |
| batch\_size      | `int`       | `20000` | Sample size per batch in stochastic training.             |

---

## 🚀 Usage Examples

```python
from soft_clustering import NOCD
import numpy as np
from scipy.sparse import csr_matrix

# Create sample graph
adjacency_matrix = csr_matrix(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
)
feature_matrix = csr_matrix(np.random.rand(3, 5))
K = 2  # target communities

# Initialize & fit
model = NOCD(random_state=42)
memberships = model.fit_predict(adjacency_matrix, feature_matrix, K)
print("\nMembership strengths per node:\n", memberships)
```

---

## 🛠️ Methods

### `fit_predict(adjacency_matrix, feature_matrix, K)`

Train the NOCD model on provided graph data and return the predicted membership matrix.

**Parameters:**

* `adjacency_matrix` (`scipy.sparse`, shape `(n_nodes, n_nodes)`): Sparse graph adjacency.
* `feature_matrix` (`scipy.sparse`, shape `(n_nodes, n_features)`): Node attribute matrix.
* `K` (`int`): Number of communities.

**Returns:**

* `memberships` (`np.ndarray`, shape `(n_nodes, K)`): Community membership degrees.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/main/soft_clustering/_nocd.py#L478)

---

## 📝 Implementation Notes

* **Undirected Graphs:** Assumes symmetry in adjacency during normalization.
* **Windows Caveat:** Wrap `NOCD.fit_predict()` calls in `if __name__ == "__main__"` to avoid multi-processing issues.

---

## 📚 Reference

1. Shchur, O., & Günnemann, S. (2019). *Overlapping Community Detection with Graph Neural Networks*. arXiv preprint [1909.12201](https://arxiv.org/abs/1909.12201).
