### Similarity-Based Soft Clustering (SISC)

> **An efficient soft clustering algorithm that allows documents to be a member of multiple clusters.**

-----

### 🔍 Overview

The **Similarity-based Soft Clustering (SISC)** algorithm is an efficient approach to identifying overlapping themes in a collection of documents. Unlike hard clustering methods, SISC enables a single document to belong to multiple clusters, which is vital for capturing documents with multi-topic content. The algorithm is particularly suitable for dynamic clustering applications, such as organizing search results. SISC works by selecting representative documents as **cluster centroids** and iteratively refining and merging these clusters based on a given similarity measure.

-----

### ⚙️ Class Definition

```python
class SISC(k: int)
```

-----

### 📋 Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `k` | `int` | None | The user-expected number of final clusters. The algorithm uses this value to initialize with `2*k` centroids for robustness. |

-----

### 🚀 Usage Examples

```python
from sisc import SISC
from scipy.sparse import csr_matrix

# Create a sample collection of documents
documents = [
    "the cat sat on the mat",
    "a dog ate my homework",
    "the cat and the dog are friends",
    "my homework is about machine learning",
    "machine learning models need data",
    "cat and dog are common pets"
]

# Initialize the SISC model with an expected number of clusters (e.g., k=2)
# The algorithm will dynamically determine the final number of clusters.
model = SISC(k=2)

# Run the clustering algorithm
memberships = model.fit_predict(documents)

# Print the resulting sparse membership matrix
print("Membership matrix:\n", memberships.toarray())
```

-----

### 🛠️ Methods

#### `fit_predict(docs)`

Runs the SISC clustering algorithm on the provided documents and returns the membership matrix. This is the main public method to call.

**Parameters:**

  * `docs` (`List[str]`): A list of text documents to be clustered.

**Returns:**

  * `memberships` (`scipy.sparse.csr_matrix`, shape `(n_docs, n_clusters)`): A sparse matrix where a non-zero element indicates a document's membership in a cluster.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_sisc.py#L252)
-----

### 📝 Implementation Notes

  * **Similarity Measure:** The algorithm uses the Tanimoto similarity coefficient, a measure of similarity between two sets. This is a crucial component as it drives the clustering process.
  * **Dynamic Thresholding:** Instead of requiring a user-defined similarity threshold, SISC calculates a dynamic `lambda` value based on the document set itself. This makes the algorithm adaptive and robust across different datasets.
  * **Randomization:** To improve efficiency, SISC employs a randomization technique during the iterative refinement phase. It only recalculates the membership measure for a document-centroid pair with a probability proportional to their current similarity, significantly reducing computation time without sacrificing cluster quality.

## 📚 Reference

1. King-Ip Lin, Ravikumar Kondadadi. *A SIMILARITY-BASED SOFT CLUSTERING ALGORITHM FOR DOCUMENTS*.(https://www.comp.nus.edu.sg/~lingtw/dasfaa_proceedings/dasfaa2001/00916362.pdf).
