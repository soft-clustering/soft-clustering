<p align="center">
  <img src="https://raw.githubusercontent.com/soft-clustering/soft-clustering/refs/heads/main/SCPP_Poster.png" alt="SCPP — Soft Clustering Python Package" width="720">
</p>

<h1 align="center">SCPP · Soft Clustering Python Package</h1>

<p align="center">
  <b>42 soft, fuzzy, possibilistic, evidential, graph, document and deep clustering algorithms — behind one estimator protocol.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/soft-clustering/"><img alt="PyPI" src="https://img.shields.io/pypi/v/soft-clustering?color=1f6feb&label=pypi"></a>
  <a href="https://pypi.org/project/soft-clustering/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/soft-clustering?color=1f6feb"></a>
  <a href="https://github.com/soft-clustering/soft-clustering/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/soft-clustering/soft-clustering/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://soft-clustering.readthedocs.io/en/latest/"><img alt="Docs" src="https://readthedocs.org/projects/soft-clustering/badge/?version=latest"></a>
  <a href="https://github.com/soft-clustering/soft-clustering/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/soft-clustering?color=green"></a>
  <a href="https://arxiv.org/abs/2607.19620"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2607.19620-b31b1b.svg"></a>
</p>

---

Hard clustering forces every point into exactly one group. Real data rarely
cooperates: a document belongs to several topics, a node joins several
communities, a cell sits between two types. **Soft clustering** assigns degrees
of membership instead — and the methods that do it are scattered across decades
of papers, each with its own interface, notation and output convention.

SCPP collects 42 of them into one library, behind one protocol, with a
benchmarking suite that lets you compare them on equal terms.

## ✨ Highlights

| | |
| --- | --- |
| 🧩 **One protocol, 42 algorithms** | Every estimator exposes `memberships_`, `labels_`, `centers_` and `n_clusters` after `fit`, whatever the method consumes — a feature matrix, a graph, raw documents, or an ensemble of partitions. |
| 🔬 **Checked, not asserted** | A conformance suite fits **all 42** estimators on every commit and verifies the contract: membership shape, the partition constraint where the formulation imposes one, out-of-sample behaviour, `sklearn.clone` round-tripping, and reproducibility under a fixed `random_state`. Exclusions are a hard CI failure. |
| 📊 **Benchmarking included** | Runtime, memory, scalability and clustering quality, over 20 datasets and 12 validity metrics — shipped *inside* the package, and actually run: see [`benchmarks/`](benchmarks/) for the cross-family comparison and the agreement checks against `scikit-fuzzy` and `scikit-learn`. |
| 🪶 **Light by default** | A base install is `numpy`, `scipy`, `scikit-learn` and `typeguard` — enough to fit 38 of the 42 estimators. PyTorch and pandas live behind extras you opt into. |
| 🏷️ **Typed** | Ships `py.typed`; hints are checked at runtime by `typeguard` and visible to your type checker. |

## 📦 Installation

```bash
pip install soft-clustering
```

Optional extras, separated by what they are needed **for**:

| Extra | Install | Provides | Needed for |
| --- | --- | --- | --- |
| *(base)* | `pip install soft-clustering` | numpy, scipy, scikit-learn, typeguard | Fitting any of the 38 non-deep estimators |
| `deep` | `pip install "soft-clustering[deep]"` | torch, torch_geometric | CDCGS, DMoN, NOCD, RDFKC |
| `bench` | `pip install "soft-clustering[bench]"` | pandas, psutil, tabulate | Running `soft_clustering.benchmarking` |
| `baselines` | `pip install "soft-clustering[baselines]"` | scikit-fuzzy | Agreement checks against third-party implementations |
| `docs` | `pip install "soft-clustering[docs]"` | sphinx, myst-parser | Building the documentation |
| `dev` | `pip install -e ".[dev,deep]"` | pytest, pytest-cov, matplotlib | Developing and testing |

Requires Python 3.10 or newer.

## 🚀 Quick start

```python
import numpy as np
from soft_clustering import FCM

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0, 0], 0.4, (100, 2)),
               rng.normal([4, 4], 0.4, (100, 2))])

model = FCM(n_clusters=2, random_state=0).fit(X)

model.memberships_    # (200, 2) degrees of membership, rows sum to 1
model.labels_         # (200,)   arg-max hard assignment
model.centers_        # (2, 2)   cluster prototypes
model.n_clusters      # 2        the partition actually produced
```

```text
>>> np.round(model.memberships_[:3], 3)
[[0.    1.   ]
 [0.003 0.997]
 [0.001 0.999]]
```

Every estimator follows the same shape, so swapping the method is a one-word
change:

```python
from soft_clustering import ECM, GK, PCM, RoughKMeans

for cls in (GK, PCM, ECM, RoughKMeans):
    model = cls(n_clusters=2).fit(X)
    print(f"{cls.__name__:20s} {model.memberships_.shape}")
```

`n_clusters` is accepted by every estimator regardless of the spelling used in
the original paper (`c`, `k`, `n_topics`, `n_communities`, …), and estimators
that discover the cluster count themselves report it in `n_clusters` after
fitting.

## 🧩 Algorithm catalogue

Grouped by what the algorithm consumes. ⚡ marks estimators requiring the
`deep` extra.

<details open>
<summary><b>Feature matrix</b> — 26 estimators</summary>

| | | |
| --- | --- | --- |
| `FCM` | Fuzzy C-Means | The canonical membership-based method |
| `PCM` | Possibilistic C-Means | Typicalities; rows need not sum to 1 |
| `PFCM` | Possibilistic Fuzzy C-Means | Combines memberships and typicalities |
| `GK` | Gustafson–Kessel | Adaptive per-cluster covariance |
| `ECM` | Evidential C-Means | Belief masses over sets of clusters |
| `KFCM` | Kernelized Fuzzy C-Means | Kernel-space distances |
| `SKFCM` | Spatially-Constrained Kernelized FCM | Adds an image neighbourhood term |
| `KFCCL` | Kernel-based Fuzzy Competitive Learning | Competitive-learning update |
| `CAFCM` | Collaborative Annealing FCM | Deterministic annealing |
| `CAFHFCM` | Centroid Auto-Fused Hierarchical FCM | Fuses centroids hierarchically |
| `ENTROPYFCM` | Entropy c-Means | Entropy regularisation in place of the fuzzifier |
| `AFCM` | AFCM with full graph embedding | Graph-regularised memberships |
| `AFCMSimple` | AFCM without graph embedding | The unregularised variant |
| `AFCMAdaptive` | Adaptive FCM for image segmentation | Operates on a single image |
| `FCC` | Fuzzy Color Clustering | Fuzzy colour spheres in CIELAB space |
| `RPFKM` | Robust Projected Fuzzy K-Means | Joint dimensionality reduction for noisy, high-dimensional data |
| `SFCMEP` | Semi-supervised Fuzzy Clustering with Membership Prior | Consumes a partially labelled target vector |
| `FeMIFuzzy` | Federated Multiple Imputation Fuzzy Clustering | Consumes per-client matrices |
| `RoughKMeans` | Rough K-Means | Lower and upper approximations |
| `SCM` | Subtractive Clustering Method | Determines the cluster count itself |
| `SoftDBSCANGM` | Soft DBSCAN with Gaussian mixtures | Density-based, discovers *k* |
| `SoftKSC` | Soft Kernel Spectral Clustering | Semi-supervised; two non-parallel hyperplanes |
| `GMM` | Gaussian Mixture (EM) | Responsibilities as memberships |
| `BGMM` | Beta-Gaussian Mixture Model | Two aligned views |
| `MBMM` | Multivariate Beta Mixture Model | For data on the unit interval |
| `RDFKC` ⚡ | Robust Deep Fuzzy K-Means | Image tensors |

</details>

<details>
<summary><b>Graphs and networks</b> — 6 estimators</summary>

| | | |
| --- | --- | --- |
| `BIGCLAM` | Cluster Affiliation Model for Big Networks | Overlapping communities at scale |
| `BayesianNMF` | Bayesian NMF for overlapping communities | Automatic relevance determination |
| `MMSB` ⚡ | Mixed Membership Stochastic Blockmodel | Generative blockmodel |
| `NOCD` ⚡ | Neural Overlapping Community Detection | GNN + Bernoulli–Poisson |
| `DMoN` ⚡ | Deep Modularity Networks | Modularity-optimising pooling |
| `CDCGS` ⚡ | Community Detection via Gumbel Softmax | Differentiable assignment |

</details>

<details>
<summary><b>Documents and text</b> — 5 estimators</summary>

| | | |
| --- | --- | --- |
| `LDA` | Latent Dirichlet Allocation | Topic–word and document–topic factors |
| `PLSI` | Probabilistic Latent Semantic Indexing | Likelihood-based topic model |
| `SISC` | Similarity-Based Soft Clustering | Discovers the cluster count |
| `KMART` | Modified Fuzzy ART for documents | Adaptive resonance |
| `WBSC` | Word-Based Soft Clustering | Word-driven soft assignment |

</details>

<details>
<summary><b>Ensembles and consensus</b> — 3 estimators</summary>

| | | |
| --- | --- | --- |
| `SCSPA` | Soft CSPA | Similarity-based consensus over soft partitions |
| `SHBGF` | Soft HBGF | Bipartite graph consensus over concatenated memberships |
| `SMCLA` | Soft MCLA | Groups clusters into meta-clusters |

</details>

Each algorithm has its own documentation page with parameters, usage and the
primary reference: **[full API reference →](https://soft-clustering.readthedocs.io/en/latest/index.html)**

## 📊 Benchmarking

The benchmarking suite ships inside the package, so it is available straight
after installation — no cloning required.

```bash
pip install "soft-clustering[bench]"
```

```python
from soft_clustering import FCM, GK, PCM
from soft_clustering.benchmarking import (
    ClusteringBenchmark,
    ClusteringQualityBenchmark,
    RuntimeBenchmark,
    get_dataset,
)

X, y = get_dataset("iris")

results = ClusteringBenchmark(
    models=[
        FCM(n_clusters=3, random_state=0),
        GK(n_clusters=3, random_state=0),
        PCM(n_clusters=3, random_state=0),
    ],
    benchmarks=[RuntimeBenchmark(n_repeats=3), ClusteringQualityBenchmark()],
).run(X, y)
```

```text
              model  fit_time_sec   ari   nmi  partition_coefficient
        FuzzyCMeans         0.001 0.729 0.750                  0.783
    GustafsonKessel         0.009 0.743 0.758                  0.727
PossibilisticCMeans         0.003 0.531 0.696                  0.256
```

| Backend | Measures |
| --- | --- |
| ⏱️ `RuntimeBenchmark` | Fit and predict time, with repeats and standard deviation |
| 💾 `MemoryBenchmark` | Resident set size sampled *during* the fit, so transient allocations are caught |
| 📈 `ScalabilityBenchmark` | Runtime and memory as the sample count grows |
| 🎯 `ClusteringQualityBenchmark` | Silhouette, Calinski–Harabasz, Davies–Bouldin; ARI and NMI when labels are given; partition coefficient and entropy |

Also included: **20 datasets** (bundled, synthetic and OpenML) via
`get_dataset`, and **12 validity metrics** usable on their own — partition
coefficient, modified partition coefficient, partition entropy, Xie–Beni,
fuzzy compactness and separation, plus the standard hard-clustering indices.

**[Benchmarking guide →](https://soft-clustering.readthedocs.io/en/latest/benchmarking.html)**

## 📚 Documentation

- **[Documentation site](https://soft-clustering.readthedocs.io/en/latest/index.html)** — API reference, one page per algorithm, benchmarking guide
- **[`example/`](https://github.com/soft-clustering/soft-clustering/tree/main/example)** — a runnable script for every estimator
- **[`tests/`](https://github.com/soft-clustering/soft-clustering/tree/main/tests)** — including the shared protocol conformance suite

## 🧪 Development

```bash
git clone https://github.com/soft-clustering/soft-clustering.git
cd soft-clustering
pip install -e ".[dev,deep]"

pytest                                                    # tests
pytest --cov=soft_clustering --cov-report=term-missing    # with coverage
ruff check soft_clustering tests example tools            # lint
black --check soft_clustering tests example tools         # formatting
sphinx-build -b html -W docs/source docs/_build/html      # docs, warnings are errors
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the estimator protocol and what
adding an algorithm involves.

## 📖 Citation

This package accompanies a paper submitted to the JMLR MLOSS track.

- **Paper:** [arXiv:2607.19620](https://arxiv.org/abs/2607.19620)

```bibtex
@misc{rezaee2026scppunifiedpythonlibrary,
   title={SCPP: A Unified Python Library for Soft Clustering},
   author={Kiyan Rezaee and Morteza Ziabakhsh and Artin Bahrampour and
           Seyed Mohammad Ghoreishi and Asal Khaje and Ali Sajedifar and
           Manny Chalak and Ava Zerafatangiz and Sadegh Eskandari},
   year={2026},
   eprint={2607.19620},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2607.19620},
}
```

GitHub's **Cite this repository** button reads
[`CITATION.cff`](CITATION.cff) and will produce this for you.

## 🤝 Contributing

Contributions, bug reports and algorithm proposals are all welcome.

- [Contributing guidelines](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](.github/SECURITY.md) — please report vulnerabilities privately
- [Open an issue](https://github.com/soft-clustering/soft-clustering/issues/new/choose)

## ⚖️ License

Distributed under the terms of the [MIT license](LICENSE).
