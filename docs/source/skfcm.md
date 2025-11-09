# SKFCM (Spatially-Constrained Kernelized Fuzzy C-Means) Documentation

---

## 🔍 Overview
SKFCM extends Kernelized Fuzzy C-Means (KFCM) by adding a spatial constraint that incorporates local neighborhood information into the clustering process. This makes it more robust to noise and suitable for image segmentation tasks like MRI.

---


## ⚙️ Class Definition
**Class Name:** `SKFCM`

```python
class SKFCM:
    def __init__(self, n_clusters: int = 3, m: float = 2.0, gamma: float = 1.0,
                 lambda_: float = 0.5, max_iter: int = 100, tol: float = 1e-5):
        ...
```

---
## 📋 Parameters

| Parameter     | Type   | Default | Description                                              |
|---------------|--------|---------|----------------------------------------------------------|
| `n_clusters`  | int    | 3       | Number of clusters                                       |
| `m`           | float  | 2.0     | Fuzziness degree                                         |
| `gamma`       | float  | 1.0     | Kernel coefficient for RBF                               |
| `lambda_`     | float  | 0.5     | Spatial constraint weight                                |
| `max_iter`    | int    | 100     | Maximum number of iterations                             |
| `tol`         | float  | 1e-5    | Tolerance for convergence                                |

---
## 🚀 Usage Examples

```python
from soft_clustering._skfcm._skfcm import SKFCM
import numpy as np

image = np.zeros((50, 50))
image[:25, :] = 0.2
image[25:, :] = 0.8
image += np.random.normal(0, 0.05, image.shape)

X = image.reshape(-1, 1)
shape = image.shape

model = SKFCM(n_clusters=2, gamma=5.0, lambda_=0.8)
model.fit(X, shape)

labels = model.predict().reshape(shape)
membership = model.predict_proba().reshape(shape[0], shape[1], -1)
```

---
### 📥 Input / 📤 Output

- **Input to `fit(X, shape)`**:
  - `X (np.ndarray)`: Flattened image data (N x 1)
  - `shape (Tuple[int, int])`: Original image shape (height, width)

- **Returns**:
  - `predict()` → Hard labels (H x W)
  - `predict_proba()` → Membership matrix reshaped to (H x W x C)

---
## 🛠️ Methods

- `fit(X, shape)`: Performs spatially-regularized clustering
- `predict()`: Returns hard cluster labels
- `predict_proba()`: Returns fuzzy membership degrees

---



## 📝 Implementation Notes

- Uses RBF kernel to capture nonlinear structure
- Adds neighborhood averaging term via uniform filter
- More robust to noise and boundary fluctuations than KFCM
- Especially effective for image segmentation

---

### 📚 Reference

This implementation is based on:  
**"A novel kernelized fuzzy C-means algorithm with application in medical image segmentation"**  
by S. Chen and D. Zhang.
