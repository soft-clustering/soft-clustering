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

[🔗 Source on GitHub](https://github.com/soft-clustering/soft-clustering/blob/main/soft_clustering/_lda.py#L7)

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


