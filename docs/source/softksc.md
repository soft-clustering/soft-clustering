# SoftKSC (Soft Kernel Spectral Clustering) Documentation

---

## 🔍 Overview
SoftKSC is a semi-supervised kernel spectral clustering algorithm that separates data using two non-parallel hyperplanes. It combines kernel learning with soft clustering logic by modeling confidence scores for both classes.

---

## ⚙️ Class Definition
**Class Name:** `SoftKSC`
This class implements the Soft Kernel Spectral Clustering using a dual-space solution and RBF kernel similarity.

```python
class SoftKSC:
    def __init__(self, gamma: float = 1.0, C: float = 1.0):
        ...
```


---

## 📋 Parameters

| Parameter | Type   | Default | Description                                 |
|-----------|--------|---------|---------------------------------------------|
| `gamma`   | float  | 1.0     | RBF kernel coefficient                       |
| `C`       | float  | 1.0     | Regularization term in the dual formulation |

---
## 🚀 Usage Examples

```python
from soft_clustering._soft_ksc._soft_ksc import SoftKSC
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
y = np.where(y == 0, -1, 1)

# Use 20% labeled, 80% unlabeled
X_labeled = X[:40]
y_labeled = y[:40]
X_unlabeled = X[40:]

model = SoftKSC(gamma=2.0, C=1.0)
model.fit(X_labeled, y_labeled, X_unlabeled)

print("Predicted Labels:", model.predict(X))
print("Soft Probabilities:", model.predict_proba(X))
```

---
### 📥 Input / 📤 Output

- **Input to `fit(X_labeled, y_labeled, X_unlabeled)`**:
  - `X_labeled (np.ndarray)`: Labeled training data (N_l x D)
  - `y_labeled (np.ndarray)`: Class labels in {-1, 1} (N_l,)
  - `X_unlabeled (np.ndarray)`: Unlabeled data (N_u x D)

- **Returns**:
  - Soft membership probabilities: `predict_proba(X)` → (N x 2)
  - Hard cluster predictions: `predict(X)` → (N,)

---

## 🛠️ Methods

- `fit(X_labeled, y_labeled, X_unlabeled)`: Fits the model using both labeled and unlabeled data
- `predict_proba(X)`: Returns soft membership scores to both classes
- `predict(X)`: Returns predicted class labels {-1, 1}

---



## 📝 Implementation Notes

- Uses RBF kernel similarity matrix for dual formulation
- Solves two linear systems for class-specific projections
- Uses relative distance to compute soft assignments
- Designed for semi-supervised clustering with partial labels

---

### 📚 Reference

This implementation is based on:  
**"Soft Kernel Spectral Clustering"**  
by Faraki et al.
