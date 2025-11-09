# FeMIFuzzy (Federated Multiple Imputation Fuzzy Clustering) Documentation

---

## 🔍 Overview
FeMIFuzzy is a federated fuzzy clustering algorithm designed for incomplete longitudinal behavioral data. It combines multiple imputation, Sammon mapping (via dimensionality reduction), and fuzzy c-means clustering, allowing decentralized learning from incomplete data.

---

## ⚙️ Class Definition
**Class Name:** `FeMIFuzzy`

```python
class FeMIFuzzy:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, m: float = 2.7, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.tol = tol
```

---

## 📋 Parameters

| Parameter       | Type   | Default | Description                                                        |
|------------------|--------|---------|--------------------------------------------------------------------|
| `n_clusters`     | int    | 3       | Number of fuzzy clusters                                           |
| `max_iter`       | int    | 100     | Maximum number of iterations                                       |
| `m`              | float  | 2.7     | Fuzziness degree (entropy-like parameter)                          |
| `tol`            | float  | 1e-4    | Convergence threshold for membership updates                       |

---

## 🚀 Usage Examples

```python
from soft_clustering._femifuzzy import FeMIFuzzy
import numpy as np

X = np.random.rand(100, 5)
X[X < 0.1] = np.nan  # Simulate missing data

model = FeMIFuzzy(n_clusters=3)
U = model.fit_predict(X)

print("Membership matrix shape:", U.shape)
```

---
### 📥 Input / 📤 Output

- **Input to `fit_predict(X)`**:
  - `X (np.ndarray)`: Input matrix with shape (N x D), possibly containing NaN values

- **Returns**:
  - `U (np.ndarray)`: Membership matrix (N x C)

---

## 🛠️ Methods

- `fit_predict(X)`: Runs the complete FeMIFuzzy pipeline:
  - Mean imputation of missing data
  - PCA (as a surrogate to Sammon mapping)
  - Iterative fuzzy c-means with convergence criteria

---



# 📝 Implementation Notes

- Imputation uses mean strategy per feature
- PCA replaces Sammon mapping for simplicity
- Handles missing values before clustering
- Memberships are normalized row-wise

---

### 📚 Reference

This implementation is based on:  
**"Federated Fuzzy Clustering for Decentralized Incomplete Longitudinal Behavioral Data"**  
by Mohammad Mahdavi, Raheleh Salari, and Jie Xu.
