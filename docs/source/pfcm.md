#  Possibilistic Fuzzy C-Means (PFCM)

> **A robust soft clustering algorithm combining fuzzy memberships and outlier-aware typicalities.**

---

## 🔍 Overview

The **Possibilistic Fuzzy C-Means (PFCM)** algorithm clusters data by assigning both:
- **Fuzzy memberships** to model soft belonging across multiple clusters
- **Possibilistic typicalities** to identify and suppress noisy or atypical points

This dual mechanism improves on standard FCM by being more **robust to noise** and avoiding **cluster collapse**, making it ideal for ambiguous data and real-world use.

---

## ⚙️ Class Definition

```python
class soft_clustering.PFCM(
    n_clusters: int,
    m: float = 2.0,
    eta: float = 2.0,
    a: float = 1.0,
    b: float = 1.0,
    max_iter: int = 150,
    tol: float = 1e-5,
    random_state: Optional[int] = None
)
```

---

## 📋 Parameters

| Parameter      | Type            | Default | Description                                         |
| -------------- | --------------- | ------- | --------------------------------------------------- |
| `n_clusters`   | `int`           | —       | Number of clusters to form.                         |
| `m`            | `float`         | `2.0`   | Fuzzifier for membership values.                    |
| `eta`          | `float`         | `2.0`   | Fuzzifier for typicality values.                    |
| `a`            | `float`         | `1.0`   | Weight for membership influence in centroid update. |
| `b`            | `float`         | `1.0`   | Weight for typicality influence in centroid update. |
| `max_iter`     | `int`           | `150`   | Maximum number of iterations.                       |
| `tol`          | `float`         | `1e-5`  | Tolerance for convergence.                          |
| `random_state` | `Optional[int]` | `None`  | Seed for reproducible centroid initialization.      |

---

## 🚀 Usage Example

```python
from soft_clustering import PFCM

# Create a simple dataset
data = [[1.0, 1.1], [0.9, 0.95], [5.1, 5.2], [5.0, 5.1], [9.0, 1.0]]

# Initialize and train the model
model = PFCM(n_clusters=3, random_state=0)
model.fit(data)

# Access results
print("Cluster centers:", model.V)
print("Memberships:", model.U)
print("Typicalities:", model.T)
```
---

## 🛠️ Methods

### `fit(X)`
Train the model on input data.

#### Parameters:

- `X` `(Union[np.ndarray, List[List[float]]])`: **Data matrix with shape `(n_samples, n_features)`**.

#### Returns:

- `self`: **Trained model instance**.



### `predict_typicalities(X)`
Compute typicalities for new data points.

#### Returns:

- `U` (`np.ndarray`, shape `(s,n_samples)`): **Membership Matrix**.



### `predict_typicalities(X)`
Compute typicalities for new data points.

#### Returns:

- `T` (`np.ndarray`, shape `(c, n_samples)`): **Typicality matrix**.

---

## 🧪 Test Example

The repository includes a simple test script at `test/test_pfcm.py`. This file:

- Generates a basic 2D dataset with 3 clusters

- Trains the PFCM model

- Prints membership and typicality results

- Optionally plots fuzzy clusters and outliers (if `matplotlib` is available)

#### Run it:

```python
python tests/test_pfcm.py
```

---

## 📊 Example Output

If you have `matplotlib` installed, you'll see:

- **Left panel**: PFCM cluster results (colored by fuzzy membership)

- **Right panel**: Outlier scores (colored by lowest typicality)

---

## 📝 Notes

- Suitable for small and medium datasets.

- Typicalities highlight noise and are useful for filtering outliers.

- Combines the best of fuzzy logic and possibilistic reasoning.

---

## 📚 Reference

1. Nikhil R. Pal, Kuhu Pal, James M. Keller, and James C. Bezdek (2005). *A Possibilistic Fuzzy C-Means Clustering Algorithm*, IEEE Transactions on Fuzzy Systems, [DOI:10.1109/TFUZZ.2004.840845](https://ieeexplore.ieee.org/document/1492404).