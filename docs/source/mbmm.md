#  MBMM (Multivariate Beta Mixture Model) Documentation

---

## 🔍 Overview
MBMM is a probabilistic soft clustering algorithm using mixtures of multivariate Beta distributions. Each feature is modeled independently with a Beta distribution. It is especially effective when features are constrained to the range (0,1).

---

## ⚙️ Class Definition
**Class Name:** `MBMM`
This class implements the EM algorithm for estimating parameters of Beta mixtures across multiple dimensions.
```python
class MBMM:
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-5):
        ...
```

---

## 📋 Parameters

| Parameter       | Type   | Default | Description                                             |
|----------------|--------|---------|---------------------------------------------------------|
| `n_components` | int    | 3       | Number of clusters/components in the mixture            |
| `max_iter`     | int    | 100     | Maximum number of iterations in the EM algorithm        |
| `tol`          | float  | 1e-5    | Convergence threshold on log-likelihood                 |

---

## 🚀 Usage Examples

```python
from soft_clustering._mbmm._mbmm import MBMM
import numpy as np

# Create synthetic beta-distributed data
X1 = np.random.beta(2, 5, size=(50, 2))
X2 = np.random.beta(5, 2, size=(50, 2))
X = np.vstack([X1, X2])

model = MBMM(n_components=2, max_iter=100)
model.fit(X)

print("Labels:", model.predict())
print("Memberships:", model.predict_proba())
```

---

### 📥 Input / 📤 Output

- **Input to `fit(X)`**:
  - `X (np.ndarray)`: Input data (N x D), all features must be in the range (0,1)

- **Returns**:
  - Membership probabilities: `predict_proba()` → array of shape (N x K)
  - Hard labels: `predict()` → array of shape (N,)

---

## 🛠️ Methods

- `fit(X)`: Fits the MBMM model using EM algorithm
- `predict_proba()`: Returns the membership probability matrix
- `predict()`: Returns hard cluster labels

---



## 📝 Implementation Notes

- Each dimension is modeled independently with a Beta distribution
- Parameters estimated via method of moments in M-step
- Log-likelihood used for convergence checking
- Input data must be scaled to (0,1)

---

### 📚 Reference

1. Kim, K., & Tewari, A. (2024). *Multivariate Beta Mixture Model: Probabilistic Clustering with Flexible Cluster Shapes*.

