### A Modified Fuzzy ART for Soft Document Clustering (KMART)

> **An efficient soft clustering algorithm that allows documents to be a member of multiple clusters.**

-----

### 🔍 Overview

The **Modified Fuzzy ART (KMART)** algorithm is an efficient approach to identifying overlapping themes in a collection of documents. Unlike hard clustering methods, KMART enables a single document to belong to multiple clusters, which is vital for capturing documents with multi-topic content. The algorithm modifies the Fuzzy ART neural network to make it more suitable for document clustering by eliminating an expensive iterative search process and dynamically determining the number of clusters.

-----

### ⚙️ Class Definition

```python
class KMART(vigilance_param: float, learning_rate: float)
```

-----

### 📋 Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vigilance_param` | `float` | `0.5` | The vigilance parameter, $\\rho$, between 0 and 1. A higher value leads to more specific (tighter) clusters. |
| `learning_rate` | `float` | `1.0` | The learning rate, $\\lambda$, between 0 and 1. If set to 1.0 (fast learning), the prototype is immediately updated to match the input. |

-----

### 🚀 Usage Examples

```python
from kmart import KMART
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

# Initialize the KMART model with default parameters
model = KMART()

# Run the clustering algorithm
memberships = model.fit_predict(documents)

# Print the resulting sparse membership matrix
print("Membership matrix:\n", memberships.toarray())
```

-----

### 🛠️ Methods

#### `fit_predict(docs)`

Runs the KMART clustering algorithm on the provided documents and returns the membership matrix. This is the main public method to call.

**Parameters:**

  * `docs` (`List[str]`): A list of text documents to be clustered.

**Returns:**

  * `memberships` (`scipy.sparse.csr_matrix`, shape `(n_docs, n_clusters)`): A sparse matrix where a non-zero element indicates a document's membership in a cluster.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_kmart.py#L132)
-----

### 📝 Implementation Notes

  * **Fuzzy ART Modification:** KMART's core innovation is its modification of the Fuzzy ART neural network. Instead of iteratively searching for the best-matching cluster, it checks all clusters against a vigilance test. This not only enables soft clustering but also significantly improves performance by removing a computationally expensive step.
  * **Dynamic Cluster Creation:** The number of clusters is not a user-defined input. New clusters are created dynamically whenever a document fails the vigilance test for all existing clusters.
  * **Keyword Extraction:** After clustering, the algorithm extracts representative keywords for each cluster, filtering out common stop words to ensure the keywords are meaningful.

## 📚 Reference

1.Ravikumar Kondadadi and Robert Kozma. *Overlapping Community Detection with Graph Neural Networks*.(http://techlab.bu.edu/files/resources/articles_tt/A%20Modified%20Fuzzy%20ART%20for%20Soft%20Document%20Clustering.pdf).
