#  Robust deep fuzzy 𝐾-means clustering for image data (RD-FKC)

> **End-to-end deep fuzzy clustering with Laplacian regularization and adaptive robustness.**

---

## 🔍 Overview

The **Robust deep fuzzy 𝐾-means clustering (RD-FKC)** algorithm combines deep representation learning and fuzzy clustering into a unified framework. It uses an encoder-decoder neural network to learn low-dimensional latent features from input images, while simultaneously optimizing cluster membership and cluster centers using an iterative strategy.


---

## ⚙️ Class Definition

```python
class soft_clustering.RDFKC(
    K: int = 10,
    encoder: Optional[torch.nn.Module] = None,
    decoder: Optional[torch.nn.Module] = None,
    dataset: Optional[str] = None,         
    random_state: Optional[int] = None,
    max_iter: int = 100,
    batch_size: Optional[int] = None,
    lr: float = 1e-4,
    mu: float = 1.0,
    gamma: float = 1e-4,
    tau: float = 0.1):
```

[🔗 Source on GitHub]()

---

## 📋 Parameters

| Parameter        | Type        | Default | Description                                                                     |
| ---------------- | ----------- | ------- | ------------------------------------------------------------------------------- |
| K                | `int`       |    –    | Number of fuzzy clusters.                                                       |
| encoder          | `nn.Module` | `None`  | Encoder model for feature learning. Required if `dataset` is not specified.     |
| decoder          | `nn.Module` | `None`  | Decoder model for reconstruction. Required if `dataset` is not specified.       |
| dataset          | `str`       | `None`  | If provided (`'coil20'` or `'fashion'`), uses default encoder/decoder pair.     |
| random\_state    | `int`       | `None`  | Seed for reproducible experiments.                                              |
| max\_iter        | `int`       | `100`   | Maximum number of update iterations.                                            |
| batch\_size      | `int`       | `None`  | Batch size for encoding. Auto-set based on dataset size if not provided.        |
| lr               | `float`     | `1e-4`  | Learning rate for Adam optimizer.                                               |
| mu               | `float`     | `1.0`   | Laplacian regularization weight.                                                |
| gamma            | `float`     | `1e-4`  | Weight regularization for encoder/decoder parameters.                           |
| tau              | `float`     | `0.1`   | Robustness coefficient for adaptive clustering loss.                            |

---

## 🚀 Usage Examples

```python
import numpy as np
from rdfkc import RDFKC

# Set seed for reproducibility
np.random.seed(42)

# Create 100 grayscale images of size 32x32 (shape: N, C, H, W)
images = np.random.rand(100, 1, 32, 32).astype(np.float32)

# Initialize RDFKC model with 5 clusters
model = RDFKC(K=5, dataset="coil20", max_iter=5)

# Fit the model and predict cluster assignments
cluster_labels = model.fit_predict(images)

# Display the number of unique clusters assigned
print(f"Found {len(np.unique(cluster_labels))} clusters.")

```

---

## 🛠️ Methods

### `fit_predict(adjacency_matrix)`

Train the RDFKC model on input images and return soft cluster assignments.

**Parameters:**

* `X` (`np.ndarray` or `torch.Tensor`, shape `(N, C, H, W)`): Input image data.


**Returns:**

* `labels` (`np.ndarray`, shape `(N,)`): Soft cluster labels (argmax of membership matrix).

[🔗 Source definition]()

---

## 📝 Implementation Notes

* **Input Format** Input X must be a NumPy array or torch Tensor of shape (N, 1, H, W) with pixel values normalized to [0, 1].
* **Iterations** Runs for a fixed max_iter; no convergence check is performed.
* **Encoder/Decoder** If dataset='coil20' or 'fashion', predefined architectures are used. Custom encoder and decoder can be provided.
 
---

## 📚 Reference

1. Wu, X., Yu, Y.-F., Chen, L., Ding, W., & Wang, Y. (2024). *Robust deep fuzzy K-means clustering for image data*.
Pattern Recognition, 153, Article 110504. (https://doi.org/10.1016/j.patcog.2024.110504).
