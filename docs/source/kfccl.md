
# Kernel-based Fuzzy Competitive Learning (K-FCCL)  
An adaptive soft-to-hard clustering algorithm using kernel-induced similarity and competitive learning dynamics.

---

## 🔍 Overview

**Kernel-based Fuzzy Competitive Learning (K-FCCL)** is a soft clustering algorithm that leverages kernel methods and competitive learning principles to uncover complex, non-linear data structures. By mapping input data into a higher-dimensional space using a Gaussian kernel, K-FCCL effectively separates clusters that are inseparable in the original feature space.

Unlike traditional fuzzy clustering methods, K-FCCL incorporates a **competitive learning mechanism** that drives the model toward optimal fuzzy partitioning using kernel-induced similarities and probabilistic softmax dynamics. This results in improved handling of overlapping clusters and more interpretable assignments.

---

## ⚙️ Class Definition

```python
class soft_clustering.KFCCL(
    n_clusters: int = 2,
    lambda_: float = 10.0,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
    max_iter: int = 100
)
```

---

## 📋 Parameters

| Parameter    | Type   | Default | Description                                                                 |
|--------------|--------|---------|-----------------------------------------------------------------------------|
| `n_clusters` | `int`  | `2`     | Number of clusters to form.                                                |
| `lambda_`    | `float`| `10.0`  | Fuzziness control parameter for softmax membership assignment. Larger values lead to crisper partitions. |
| `gamma`      | `float`| `1.0`   | Parameter for the Gaussian (RBF) kernel controlling how tightly the similarity decays. Must be tuned for performance. |
| `epsilon`    | `float`| `1e-4`  | Convergence tolerance. Iterations stop when inner products stabilize below this threshold. |
| `max_iter`   | `int`  | `100`   | Maximum number of training iterations.                                     |

---

## 🛠️ Methods

### `fit(X)`

Trains the K-FCCL model on the input dataset using kernel similarity and iterative soft competition.

**Parameters**:  
- `X` (`np.ndarray`): Data matrix of shape `(n_samples, n_features)`.

**Returns**:  
- `labels` (`np.ndarray`): Final hard cluster assignments computed from the soft membership matrix `U`.

---

## 📌 Key Features

- ✅ **Kernelized Learning**: Uses Gaussian RBF kernel to model non-linear cluster boundaries.
- ✅ **Softmax-based Fuzziness**: Learns fuzzy memberships using a temperature-scaled softmax driven by inner product similarities.
- ✅ **Online Competitive Update Rule**: Incorporates competitive learning with adaptive learning rate for more robust convergence.
- ✅ **Supports Non-Convex Clusters**: Especially effective when clusters are non-spherical or not linearly separable.

---

## 📝 Notes

- The **`gamma`** parameter has a major impact on clustering quality. Smaller values produce more global similarity, while larger values focus on local structure.
- **Fuzziness (`lambda_`)** affects how strongly each data point is assigned to its clusters. Higher values encourage hard assignments.
- The method returns **hard labels**, but the full fuzzy partitioning is available via the `U` matrix if needed.
- Use visual inspection and metrics like Adjusted Rand Index (ARI) for evaluating clustering quality, especially on synthetic or labeled data.

---

## 📚 Reference

K. Mizutani, S. Miyamoto: *"Kernel-Based Fuzzy Competitive Learning Clustering" ieeexplore.ieee.org/document/1452468*

---
 