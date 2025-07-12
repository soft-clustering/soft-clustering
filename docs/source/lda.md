#  Latent Dirichlet Allocation (LDA)

>**This module implements Latent Dirichlet Allocation (LDA) using variational EM inference.**

---

## 🔍 Overview

**Latent Dirichlet Allocation (LDA)** is a generative probabilistic model for collections of discrete data such as text corpora. It discovers hidden topic structures in the documents, with each topic being a distribution over words, and each document a distribution over topics.

---

## ⚙️ Class Definition

```python
class soft_clustering.LDA(
    n_topics: int = 10,
    alpha: float = None,
    beta: float = 0.01,
    max_iter: int = 100,
    var_max_iter: int = 20,
    tol: float = 1e-4)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_lda.py#L8)

---

## 📋 Parameters

| Parameter     | Type    | Default | Description                                       |
| ------------- | ------- | ------- | ------------------------------------------------- |
| n\_topics     | `int`   | `10`    | Number of latent topics in the corpus.            |
| alpha         | `float` | `None`  | Prior for document-topic distribution.            |
| beta          | `float` | `0.01`  | Prior for topic-word distribution.                |
| max\_iter     | `int`   | `100`   | Maximum EM iterations.                            |
| var\_max\_iter| `int`   | `20`    | Max variational steps per document.               |
| tol           | `float` | `1e-4`  | Convergence threshold for stopping EM iterations. |

---

## 🚀 Usage Examples

```python
from soft_clustering import LDA

# Sample documents
docs = [
    "apple banana apple",
    "banana fruit apple",
    "fruit banana banana"
]

# Initialize and fit the model
model = LDA(n_topics=2, max_iter=20, var_max_iter=10)
model.fit(docs)

# Print top words in each topic
model.print_top_words(n_top_words=5)
```

---

## 🛠️ Methods

### `fit(X, vocabulary=None)`

Fits the LDA model to a corpus using variational EM inference.

**Parameters:**

* `X` (`list[str]` or `csr_matrix`): Input documents as raw strings or a precomputed term-document matrix.
*  `vocabulary` (`list[str]`, `optional`): Fixed vocabulary to use when building the term-document matrix.

**Returns:**

* `self` (`LDA`): The trained model instance.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_lda.py#L56)

### `get_topic_word_dist()`

Returns the normalized topic-word distribution matrix.

**Returns:**

* `topic_word` (`ndarray` of shape `(n_topics, V)`): Each row is a probability distribution over the vocabulary for a topic.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_lda.py#L132)

### `print_top_words(n_top_words=10)`

Prints the top words in each topic based on their probabilities.

**Parameters:**

* `n_top_words` (`int`): Number of top words to display per topic.

**Returns:**

* `None`

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_lda.py#L143)

---

## 📚 Reference

1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).
*Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993–1022. (https://jmlr.org/papers/v3/blei03a.html)