#  MMSB (Mixed Membership Stochastic Blockmodel) Documentation

---

## 🔍 Overview
MMSB is a probabilistic generative model for graph data where each node is represented by a distribution over latent communities (blocks). Each edge is generated based on sampled community assignments from Dirichlet-distributed memberships and a block interaction matrix B.

---

## ⚙️ Class Definition
**Class Name:** `MMSB`

This class implements a simplified version of MMSB for sampling synthetic graphs.

```python
class MMSB:
    def __init__(self, n_nodes: int, n_blocks: int, alpha: float = 0.5):
        ...
```


---

## 📋 Parameters

| Parameter     | Type   | Default | Description                                  |
|----------------|--------|---------|----------------------------------------------|
| `n_nodes`     | int    | —       | Number of graph nodes                        |
| `n_blocks`    | int    | —       | Number of latent blocks (communities)        |
| `alpha`       | float  | 0.5     | Dirichlet prior parameter for membership     |

---
### 💻 Using Example

```python
from soft_clustering._mmsb._mmsb import MMSB

model = MMSB(n_nodes=6, n_blocks=3, alpha=0.5)
Y = model.sample_graph()
pi = model.get_memberships()
B = model.get_block_matrix()

print("Adjacency Matrix:", Y)
print("Memberships:", pi)
print("Block Matrix:", B)
```

---

### 📥 Input / 📤 Output

- **Input**: None required at inference time (sampling model)
- **Returns**:
  - `Y (Tensor)`: Adjacency matrix (n_nodes x n_nodes)
  - `pi (Tensor)`: Membership matrix (n_nodes x n_blocks)
  - `B (Tensor)`: Block interaction matrix (n_blocks x n_blocks)

---

## 🛠️ Methods

- `sample_graph()`: Generates a synthetic adjacency matrix from MMSB
- `get_memberships()`: Returns Dirichlet-distributed node memberships
- `get_block_matrix()`: Returns the B matrix of interaction probabilities

---



## 📝 Implementation Notes

- Memberships are sampled from Dirichlet distributions
- Each pair of nodes samples latent community assignments
- Edge presence determined by Bernoulli trials from B matrix

---

### 📚 Reference

This implementation is based on:  
**"Mixed Membership Stochastic Blockmodels"**  
by Airoldi et al. (2008).
