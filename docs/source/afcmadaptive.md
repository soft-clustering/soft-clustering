#  AFCMAdaptive (Adaptive Fuzzy C-Means for Image Segmentation) Documentation

---
## 🔍 Overview

AFCMAdaptive is an image segmentation algorithm that extends Fuzzy C-Means by introducing a spatially varying multiplier field `m(i,j)`. This adaptive mechanism improves segmentation in images with intensity inhomogeneity such as MRI or CT scans.

---
## ⚙️ Class Definition

**Class Name:** `AFCMAdaptive`

This class implements the AFCM algorithm with iterative updates for cluster centers, fuzzy membership, and the multiplier field.

```python
class AFCMAdaptive:
    def __init__(self, n_clusters: int = 3, m: float = 2.0, k1: float = 0.1, k2: float = 0.1,
                 max_iter: int = 100, tol: float = 1e-4):
        ...
```
---
## 📋 Parameters

| Parameter     | Type   | Default | Description                                                 |
|---------------|--------|---------|-------------------------------------------------------------|
| `n_clusters`  | int    | 3       | Number of segmentation classes                              |
| `m`           | float  | 2.0     | Fuzziness degree                                            |
| `k1`          | float  | 0.1     | First-order regularization weight for multiplier field      |
| `k2`          | float  | 0.1     | Second-order regularization weight for multiplier field     |
| `max_iter`    | int    | 100     | Maximum number of EM iterations                             |
| `tol`         | float  | 1e-4    | Convergence tolerance for center updates                    |

---
## 🚀 Usage Examples

```python
from soft_clustering._afcm_adaptive._afcm_adaptive import AFCMAdaptive
import numpy as np

# Sample synthetic image
image = np.zeros((64, 64))
image[:32, :] = 0.3
image[32:, :] = 0.7
image += np.random.normal(0, 0.05, image.shape)

model = AFCMAdaptive(n_clusters=2)
model.fit(image)

labels = model.predict()
membership = model.get_membership()
```
---
### 📥 Input / 📤 Output

- **Input to `fit(image)`**:
  - `image (np.ndarray)`: 2D grayscale image with values ∈ [0,1] or scaled

- **Returns**:
  - `predict()` → 2D integer array with cluster labels (H x W)
  - `get_membership()` → 3D array with fuzzy membership values (H x W x C)

---

## 🛠️ Methods

- `fit(image)`: Performs iterative fuzzy clustering on 2D image
- `predict()`: Returns hard labels for each pixel
- `get_membership()`: Returns fuzzy membership matrix for each class

---
## 📝 Implementation Notes

- Multiplier field m(i,j) is updated based on local pixel statistics
- Uses Gaussian smoothing and Laplacian (∇m, ∇²m) for regularization
- Memberships updated using a Mahalanobis-like normalized distance
- Avoids hard intensity boundaries by spatial adaptation

---
### 📚 Reference

This implementation is based on:  
**"An adaptive fuzzy C-means algorithm for image segmentation in the presence of intensity inhomogeneity"**  

