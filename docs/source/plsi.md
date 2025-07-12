#  Probabilistic Latent Semantic Indexing (PLSI)

> **A probabilistic model for uncovering latent topics in documents using Expectation-Maximization.**

---

## 🔍 Overview

The **Probabilistic Latent Semantic Indexing (PLSI)** algorithm uncovers latent topics in text data by modeling each document as a mixture of topics and each topic as a distribution over words, using the Expectation-Maximization (EM) algorithm to iteratively refine these probabilities from word-document co-occurrence data.

---

## ⚙️ Class Definition

```python
class PLSI(
    n_topics: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
    tempered: bool = True,
    beta_start: float = 1.0,
    beta_step: float = 0.9,
    heldout_ratio: float = 0.1,
    random_state: int = None
)
```

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_plsi.py#L8)

---

## 📋 Parameters

| Parameter      | Type    | Default | Description                                                     |
|----------------|---------|---------|-----------------------------------------------------------------|
| n\_topics      | `int`   | `10`    | Number of latent topics to discover in the corpus.              |
| max\_iter      | `int`   | `100`   | Maximum number of EM (Expectation-Maximization) iterations.     |
| tol            | `float` | `1e-4`  | Convergence threshold for the change in log-likelihood.         |
| tempered       | `bool`  | `True`  | Whether to use Tempered EM for better generalization.           |
| beta\_start    | `float` | `1.0`   | Initial inverse temperature for tempered EM.                    |
| beta\_step     | `float` | `0.9`   | Multiplicative factor to decrease beta in each iteration.       |
| heldout\_ratio | `float` | `0.1`   | Fraction of word tokens held out for validation in tempered EM. |
| random\_state  | `int`   | `None`  | Seed for reproducibility of random operations.                  |

---

## 🚀 Usage Examples

```python
from soft_clustering import PLSI
from sklearn.datasets import fetch_20newsgroups

# Load a small text corpus
newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball'])
documents = newsgroups.data[:100]  # Use a subset for quick demonstration

# Initialize the PLSI model
model = PLSI(
    n_topics=5,
    max_iter=50,
    tempered=True,
    beta_start=1.0,
    beta_step=0.9,
    heldout_ratio=0.1,
    random_state=42
)

# Fit the model to the corpus
model.fit_predict(documents)

# Get topic distributions
topic_word = model.P_w_given_z        # shape: (n_topics, n_words)
document_topic = model.P_z_given_d    # shape: (n_topics, n_documents)

# Get word distribution per document
word_given_doc = model.get_P_w_given_d()  # shape: (n_documents, n_words)

# Print perplexity
print("Perplexity:", model.perplexity)
```

---

## 🛠️ Methods

### `fit_predict(data)`

Train the PLSI model on a text corpus or term-document matrix.

**Parameters:**

* `data` (`list[str]` or `scipy.sparse.csr_matrix`, shape `(n_documents, n_words)`): Raw text documents or a sparse term-document matrix.


**Returns:**

* `None`

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_plsi.py#L181)

## `get_P_w_given_d()`

Compute the word distribution for each document  based on the learned topic and document distributions.

**Parameters:**

* `None`

**Returns:**

* `word_given_doc` (`np.ndarray`, shape `(n_documents, n_words)`): Smoothed word distribution per document.

[🔗 Source definition](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_plsi.py#L252)

---

## 📝 Implementation Notes

* **No smoothing or priors used:** This implementation does not include Dirichlet priors or additive smoothing on P(w|z) or P(d|z), which may lead to zero probabilities if topics are underrepresented or vocabulary is sparse.

---

## 📚 Reference

1. Hofmann, T. (1999). *Probabilistic Latent Semantic Indexing*. Proceedings of the 22nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 50–57. [10.1145/312624.31264](https://doi.org/10.1145/312624.312649)
