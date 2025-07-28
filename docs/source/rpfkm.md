# RPFKM Algorithm Documentation

---

## 🔍 Overview
The Robust Projected Fuzzy K-Means (RPFKM) is a fuzzy clustering algorithm designed for high-dimensional data with noise and outliers. It combines fuzzy clustering with dimensionality reduction and robustness mechanisms to improve clustering quality.

---

## ⚙️ Class Definition
**Class Name:** `RPFKM`

Implements the RPFKM algorithm with projection learning and robust optimization. Exposes a `FeedPredict` method to run clustering on data.

```python
class RPFKM(
    c: int,
    d: int,
    gamma: float = 1.0,
    beta: float = 0.5,
    max_iter: int = 100
)
```

## 📋 Parameters
| Parameter   | Type    | Default | Description                                      |
|-------------|---------|---------|--------------------------------------------------|
| `c`         | `int`   | —       | Number of clusters                               |
| `d`         | `int`   | —       | Dimension of reduced subspace                    |
| `gamma`     | `float` | `1.0`   | Fuzzy membership regularization                  |
| `beta`      | `float` | `0.5`   | Data reconstruction regularization               |
| `max_iter`  | `int`   | `100`   | Number of iterations                             |

## 🚀 Usage Examples
import numpy as np
from rpfkm import RPFKM  # Assuming RPFKM class is saved in rpfkm.py

def test_rpfkm_basic():
    from sklearn.datasets import make_blobs

    # Generate synthetic dataset
    X, _ = make_blobs(n_samples=100, centers=3, n_features=10, random_state=42)
    X = X.T  # Transpose to shape (D, N)

    # Initialize and run the RPFKM algorithm
    model = RPFKM(c=3, d=5, gamma=0.1, beta=1.0, max_iter=10)
    labels, U, W = model.fit_predict(X)

    print("Cluster labels:", labels)
    print("Membership matrix U shape:", U.shape)
    print("Projection matrix W shape:", W.shape)

if __name__ == "__main__":
    test_rpfkm_basic()




## 🛠️ Methods

### `fit_predict(X)`

Cluster input data `X` using the RPFKM algorithm.

**Parameters:**

- `X` (`np.ndarray`, shape `(D, N)`): Input data with `D` features and `N` samples.

**Returns:**

- `labels` (`np.ndarray`): Cluster label per sample.
- `U` (`np.ndarray`): Fuzzy membership matrix `(c, N)`.
- `W` (`np.ndarray`): Projection matrix `(D, d)`.



## 📝 Implementation Notes

- Projection matrix `W` is learned via eigen decomposition of a scatter matrix difference.
- Membership matrix `U` is updated using softmax-like fuzzy assignment.
- Auxiliary weights `p` increase robustness against outliers.
- Initialization uses Dirichlet distribution for `U` and orthogonal random matrix for `W`.

---
## 📚 Reference

This implementation is based on the following paper:
**Improving Projected Fuzzy K-Means Clustering via Robust Learning**  


---