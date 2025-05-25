# Word-Based Soft Clustering (WBSC)

> **An efficient word-based algorithm for discovering overlapping topics in documents.**

---

## 🔍 Overview

The **Word-Based Soft Clustering (WBSC)** algorithm identifies overlapping themes in a collection of documents. It operates by first clustering the words used in the documents and then hierarchically merging clusters that contain similar sets of documents. This approach allows a single document to belong to multiple clusters, effectively capturing multi-topic documents.

---

## ⚙️ Class Definition

```python
class soft_clustering.WBSC(
    similarity_threshold: float = 0.33,
    min_doc_freq: int = 2,
    max_doc_freq_ratio: float = 0.5
)
```

---

## 📋 Parameters

| Parameter              | Type    | Default | Description                                                               |
| ---------------------- | ------- | ------- | ------------------------------------------------------------------------- |
| similarity\_threshold  | `float` | `0.33`  | Tanimoto similarity score required to merge two clusters.                 |
| min\_doc\_freq         | `int`   | `2`     | The minimum number of documents a word must appear in to be considered.   |
| max\_doc\_freq\_ratio  | `float` | `0.5`   | The maximum ratio of documents a word can appear in to be considered.     |

---

## 🚀 Usage Examples

```python
from soft_clustering import WBSC
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

# Initialize and fit the model
model = WBSC(similarity_threshold=0.2, min_doc_freq=2)

memberships = model.fit_predict(documents)
print("Membership matrix:\n", memberships.toarray())
```

---

## 🛠️ Methods

### `fit_predict(docs)`

Run the WBSC clustering algorithm on the provided documents and return the membership matrix.

**Parameters:**

* `docs` (`List[str]`): A list of documents to be clustered.

**Returns:**

* `memberships` (`scipy.sparse.csr_matrix`, shape `(n_docs, n_clusters)`): A sparse matrix where a non-zero element indicates a document's membership in a cluster.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_wbsc.py#L87)

---

## 📝 Implementation Notes

* **Tokenization:** The algorithm uses a simple, space-based tokenizer that converts text to lowercase.
* **Threshold Tuning:** The quality and granularity of the resulting clusters are highly dependent on the `similarity_threshold` parameter. Lower values will result in fewer, broader clusters, while higher values will produce more, specific clusters.

## 📚 Reference

1. King-Ip Lin, Ravikumar Kondadadi. *A WORD-BASED SOFT CLUSTERING ALGORITHM FOR DOCUMENTS*.(https://www.cs.memphis.edu/~linki/_mypaper/CATA01.doc).
