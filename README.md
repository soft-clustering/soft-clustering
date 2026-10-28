<p align="center">
  <img src="https://raw.githubusercontent.com/soft-clustering/soft-clustering/refs/heads/main/SCPP_Poster.png" alt="SCPP — Soft Clustering Python Package" width="720">
</p>

<h1 align="center">SCPP · Soft Clustering Python Package</h1>

<p align="center">
  <b>42 soft clustering algorithms — fuzzy, possibilistic, evidential, probabilistic, graph, document, ensemble and deep — behind one estimator protocol.</b>
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

Hard clustering puts every sample in exactly one group. Many data sets do not
cooperate: a document covers several topics, a node belongs to several
communities, a pixel lies on the boundary between two tissues, a measurement
is an outlier that belongs nowhere. **Soft clustering** assigns degrees of
membership instead. The methods that do so are spread across decades of
literature, each with its own interface, notation and meaning of "membership".

SCPP implements 42 of them behind one interface, documents what each one's
output means, and states how far each implementation has been verified.

## Why SCPP?

- **One protocol.** Every estimator, whatever it consumes, publishes
  `memberships_`, `labels_`, `centers_` and `n_clusters_` after `fit`, and
  `fit_predict` returns the membership matrix. Swapping methods is a one-word
  change, and evaluation code is written once.
- **Honest semantics.** "Membership" means different things: a fuzzy degree,
  a posterior probability, a possibilistic typicality, a pignistic
  probability, an unnormalised affiliation. The [algorithm table](#algorithms)
  says which, and whether rows sum to one.
- **Honest verification status.** Every estimator is labelled *validated*,
  *reference*, *adapted* or *unverified* — see [the table](#algorithms) and
  [Testing](#testing) for what each level means and how it is checked.
- **Reproducible.** Every stochastic estimator takes `random_state`; a seeded
  fit is reproducible and never touches NumPy's or PyTorch's global random
  state. Both properties are tested for every estimator.
- **Light.** The base install needs NumPy, SciPy, scikit-learn and typeguard.
  PyTorch is only needed for the four deep estimators.

SCPP is meant for researchers and practitioners who want to compare soft
clustering methods, or to use a specific one, without re-implementing it.

## Installation

```bash
pip install soft-clustering
```

| Extra | Command | Adds | Needed for |
| --- | --- | --- | --- |
| *(none)* | `pip install soft-clustering` | numpy, scipy, scikit-learn, typeguard | every estimator except the four deep ones |
| `deep` | `pip install "soft-clustering[deep]"` | torch | CDCGS, DMoN, NOCD, RDFKC |
| `bench` | `pip install "soft-clustering[bench]"` | pandas, psutil, tabulate | `soft_clustering.benchmarking` |
| `baselines` | `pip install "soft-clustering[baselines]"` | scikit-fuzzy, evclust, networkx | agreement tests against third-party code |
| `docs` | `pip install "soft-clustering[docs]"` | Sphinx and extensions | building the documentation |
| `dev` | `pip install -e ".[dev]"` | test, lint and benchmark tooling | development |

SCPP supports Python 3.10–3.14.

## Quick start

```python
import numpy as np
from soft_clustering import FCM

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0, 0], 0.6, (100, 2)),
               rng.normal([3, 3], 0.6, (100, 2))])

model = FCM(n_clusters=2, random_state=0).fit(X)

model.memberships_     # (200, 2) fuzzy memberships; each row sums to 1
model.labels_          # (200,)   arg-max hard assignment
model.centers_         # (2, 2)   cluster prototypes
model.n_clusters_      # 2

# Prototype-based estimators can assign new samples.
model.predict_proba([[1.5, 1.5]])   # about [[0.5, 0.5]]: on the boundary
```

Swapping the method changes one word, and each method's output keeps its own
meaning:

```python
from soft_clustering import ECM, GK, GMM, PCM

for cls in (GK, PCM, ECM, GMM):
    U = cls(n_clusters=2, random_state=0).fit_predict(X)
    print(f"{cls.__name__:20s} first row {U[0].round(3)}, sums to {U[0].sum():.3f}")
# PCM typicalities are not normalised: an outlier is atypical of every cluster.
```

Not every method takes a feature matrix. Graph methods take an adjacency
matrix, document methods take a list of strings, consensus methods take a list
of membership matrices:

```python
from soft_clustering import LDA, SMCLA, BayesianNMF

docs = ["fuzzy clustering membership", "graph community nodes",
        "fuzzy membership degree", "community detection graph"]
LDA(n_clusters=2, random_state=0).fit(docs).memberships_      # document-topic proportions

A = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], float)
BayesianNMF(n_clusters=2, random_state=0).fit(A).memberships_  # node memberships

U1 = FCM(n_clusters=2, random_state=0).fit_predict(X)
U2 = FCM(n_clusters=2, random_state=1, m=1.5).fit_predict(X)
SMCLA(n_clusters=2, random_state=0).fit_predict([U1, U2])       # soft consensus
```

`list_algorithms()` returns the same metadata as the table below:

```python
from soft_clustering import list_algorithms

for a in list_algorithms(family="evidential"):
    print(a.name, "-", a.method, "-", a.status)
```

## The estimator protocol

| After `fit` | Meaning |
| --- | --- |
| `memberships_` | `(n_samples, n_clusters_)` non-negative soft output; see the *Output* column below for what it means |
| `labels_` | arg-max of `memberships_`; `-1` for a sample with no membership at all (possible only when rows need not sum to one) |
| `centers_` | `(n_clusters_, n_features)` prototypes for prototype-based methods, otherwise `None` |
| `n_clusters_` | number of clusters actually produced (discovered by some methods) |

- `fit(...)` returns `self`; `fit_predict(...)` returns `memberships_`.
- `predict()` and `predict_proba()` with no argument return the fitted
  partition. With new data they apply the estimator's membership rule
  **only** for inductive estimators (FCM, GK, PCM, PFCM, GMM, Gath–Geva, ECM,
  ENTROPYFCM, KFCM, KFCCL, FCC, AFCMSimple, SCM, SoftKSC); the others are
  transductive and raise `NotImplementedError` instead of returning labels of
  the wrong data.
- `n_clusters` is the constructor argument everywhere; older spellings (`c`,
  `k`, `K`, `n_topics`, `n_components`, `n_communities`, `n_blocks`) are still
  accepted.
- scikit-learn: the inductive feature-matrix estimators (FCM, GK, GMM, PCM,
  PFCM, Gath–Geva, ECM, ENTROPYFCM, AFCMSimple, KFCM, FCC) pass
  scikit-learn's `check_estimator` suite except two sparse-input checks that
  assume `predict_proba` belongs to a classifier
  (`tests/test_sklearn_compat.py`), so they work in pipelines and with
  `clone`. Other estimators support `get_params`/`set_params`/`clone`, but
  their `fit` takes the input their method needs (a graph, documents, two
  views…). In every case `fit_predict` returns memberships, not labels; use
  `fit(X).labels_` for labels.

<a id="algorithms"></a>
## Algorithms

Generated from `soft_clustering/_registry.py`
(`python tools/algorithm_table.py --write`); a test fails if it is stale.

<!-- algorithms:start -->
| Estimator | Method | Input | Output | K | Status | Reference |
| --- | --- | --- | :---: | :---: | :---: | --- |
| **Fuzzy and kernel** | | | | | | |
| `AFCM` | Adaptive fuzzy c-means with graph embedding | feature matrix | F | given | reference | [Chen et al. (2024)](https://arxiv.org/abs/2405.13427) |
| `AFCMAdaptive` | Adaptive fuzzy c-means for images with a bias field | grayscale image | F | given | reference | [Pham and Prince (1999)](https://doi.org/10.1016/S0167-8655(98)00121-4) |
| `AFCMSimple` | Adaptive fuzzy c-means without graph embedding | feature matrix | F | given | reference | [Chen et al. (2024)](https://arxiv.org/abs/2405.13427) |
| `CAFCM` | Collaborative annealing fuzzy c-means | feature matrix | F | given | adapted | [Li and Wang (2024)](https://doi.org/10.1109/TFUZZ.2023.3319663) |
| `CAFHFCM` | Centroid auto-fused hierarchical fuzzy c-means | feature matrix | F | discovered | adapted | [Lin and Chen (2021)](https://doi.org/10.1109/TFUZZ.2020.2991306) |
| `ENTROPYFCM` | Entropy-regularised fuzzy c-means | feature matrix | F | given | reference | Li and Mukaidono (1995) |
| `FCC` | Fuzzy color clustering | feature matrix (colors) | F | given | adapted | [Kim and Lee (2024)](https://arxiv.org/abs/2407.06782) |
| `FCM` | Fuzzy c-means | feature matrix | F | given | validated | [Bezdek (1981)](https://doi.org/10.1007/978-1-4757-0450-1) |
| `FeMIFuzzy` | Federated fuzzy clustering with multiple imputation | per-client feature matrices | F | selected | adapted | [Ngo et al. (2024)](https://doi.org/10.1109/JIOT.2023.3343719) |
| `GathGeva` | Gath-Geva fuzzy maximum-likelihood clustering | feature matrix | F | given | reference | [Gath and Geva (1989)](https://doi.org/10.1016/0167-8655(89)90040-8) |
| `GK` | Gustafson-Kessel fuzzy clustering | feature matrix | F | given | validated | [Gustafson and Kessel (1979)](https://doi.org/10.1109/CDC.1978.268028) |
| `KFCCL` | Kernel-based fuzzy competitive learning | feature matrix | F | given | unverified | [Mizutani and Miyamoto (2005)](https://doi.org/10.1109/FUZZY.2005.1452468) |
| `KFCM` | Kernelised fuzzy c-means | feature matrix | F | given | reference | [Zhang and Chen (2004)](https://doi.org/10.1016/j.artmed.2004.01.012) |
| `RDFKC` ⚡ | Robust deep fuzzy k-means | feature matrix or image tensor | F | given | unverified | [Wu et al. (2024)](https://doi.org/10.1016/j.patcog.2024.110504) |
| `RPFKM` | Robust projected fuzzy k-means | feature matrix | F | given | unverified | [Zhao et al. (2022)](https://doi.org/10.1016/j.neucom.2022.03.043) |
| `SFCMEP` | Semi-supervised fuzzy clustering with a prior membership matrix | feature matrix + partial labels | F | given | unverified | [Xu et al. (2024)](https://doi.org/10.1016/j.eswa.2023.121812) |
| `SKFCM` | Spatially constrained kernelised fuzzy c-means | pixel features + image shape | F | given | reference | [Zhang and Chen (2004)](https://doi.org/10.1016/j.artmed.2004.01.012) |
| `SoftKSC` | Soft kernel spectral clustering | feature matrix | S | given | reference | [Langone et al. (2013)](https://doi.org/10.1109/IJCNN.2013.6706850) |
| **Possibilistic and evidential** | | | | | | |
| `ECM` | Evidential c-means | feature matrix | E | given | validated | [Masson and Denoeux (2008)](https://doi.org/10.1016/j.patcog.2007.08.014) |
| `EVCLUS` | Evidential clustering of proximity data | feature matrix or dissimilarities | E | given | adapted | [Denoeux and Masson (2004)](https://doi.org/10.1109/TSMCB.2002.806496) |
| `PCM` | Possibilistic c-means | feature matrix | T | given | reference | [Krishnapuram and Keller (1993)](https://doi.org/10.1109/91.227387) |
| `PFCM` | Possibilistic fuzzy c-means | feature matrix | F | given | validated | [Pal et al. (2005)](https://doi.org/10.1109/TFUZZ.2004.840099) |
| **Probabilistic, density and rough-set** | | | | | | |
| `BGMM` | Beta-Gaussian mixture over two views | a Gaussian and a Beta view | P | given | adapted | Dai et al. (2008) |
| `GMM` | Gaussian mixture model (EM) | feature matrix | P | given | validated | [Dempster et al. (1977)](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x) |
| `MBMM` | Mixture of Beta distributions | feature matrix in (0, 1) | P | given | adapted | [Hsu and Chen (2024)](https://doi.org/10.1007/978-981-97-2242-6_19) |
| `RoughKMeans` | Rough k-means | feature matrix | R | given | reference | [Lingras and West (2004)](https://doi.org/10.1023/B:JIIS.0000029668.88665.1a) |
| `SCM` | Subtractive clustering | feature matrix | F | discovered | reference | [Chiu (1994)](https://doi.org/10.3233/IFS-1994-2306) |
| `SoftDBSCANGM` | Soft DBSCAN with Gaussian-mixture refinement | feature matrix | F | discovered | unverified | [Smiti and Elouedi (2016)](https://doi.org/10.1109/IS.2016.7737459) |
| **Graphs** | | | | | | |
| `BayesianNMF` | Bayesian NMF for overlapping communities | adjacency matrix | M | given | reference | [Psorakis et al. (2011)](https://doi.org/10.1103/PhysRevE.83.066114) |
| `BIGCLAM` | Cluster affiliation model for big networks | adjacency matrix | A | given | adapted | [Yang and Leskovec (2013)](https://doi.org/10.1145/2433396.2433471) |
| `CDCGS` ⚡ | Community detection via Gumbel-softmax | adjacency matrix | S | given | adapted | [Acharya and Zhang (2020)](https://doi.org/10.1007/s42979-020-00264-2) |
| `DMoN` ⚡ | Deep modularity networks | node features + graph | S | given | adapted | [Tsitsulin et al. (2023)](https://jmlr.org/papers/v24/20-998.html) |
| `MMSB` | Mixed-membership stochastic blockmodel | adjacency matrix | M | given | reference | [Airoldi et al. (2008)](https://jmlr.org/papers/v9/airoldi08a.html) |
| `NOCD` ⚡ | Neural overlapping community detection | adjacency matrix (+ node features) | A | given | adapted | [Shchur and Gunnemann (2019)](https://arxiv.org/abs/1909.12201) |
| **Documents** | | | | | | |
| `KMART` | Modified fuzzy ART for documents | documents | D | discovered | unverified | [Kondadadi and Kozma (2002)](https://doi.org/10.1109/IJCNN.2002.1007544) |
| `LDA` | Latent Dirichlet allocation | documents or term counts | M | given | validated | [Blei et al. (2003)](https://jmlr.org/papers/v3/blei03a.html) |
| `PLSI` | Probabilistic latent semantic indexing | documents or term counts | M | given | reference | Hofmann (1999) |
| `SISC` | Similarity-based soft clustering of documents | documents | D | discovered | unverified | [Lin and Kondadadi (2001a)](https://doi.org/10.1109/DASFAA.2001.916362) |
| `WBSC` | Word-based soft clustering of documents | documents | D | discovered | unverified | Lin and Kondadadi (2001b) |
| **Ensembles** | | | | | | |
| `SCSPA` | Soft CSPA consensus | list of membership matrices | H | given | adapted | [Punera and Ghosh (2008)](https://doi.org/10.1080/08839510802170546) |
| `SHBGF` | Soft HBGF consensus | list of membership matrices | H | given | adapted | [Punera and Ghosh (2008)](https://doi.org/10.1080/08839510802170546) |
| `SMCLA` | Soft MCLA consensus | list of membership matrices | F | given | adapted | [Punera and Ghosh (2008)](https://doi.org/10.1080/08839510802170546) |

**Output** (what `memberships_` holds): **F** fuzzy membership (rows sum to one); **T** possibilistic typicality (rows need not sum to one); **P** posterior probability (rows sum to one); **E** pignistic probability of a credal partition (rows sum to one); **R** rough-set membership read off lower/upper approximations; **A** non-negative affiliation strength (unnormalised); **M** mixed-membership or topic proportion (rows sum to one); **S** soft assignment from a relaxation (rows sum to one); **D** binary degree of match; a sample may match several clusters or none; **H** crisp consensus partition, one-hot.

**Status**: **validated** — agrees with an independent implementation, or reduces exactly to one in a limit, in tests/test_external_agreement.py; **reference** — implements the update equations of the cited publication, checked against it in the 0.2.0 audit; invariants and planted-structure recovery are tested, but no independent implementation was compared; **adapted** — implements the cited method with documented deviations; see the Notes or 'Deviation' paragraphs of the estimator's docstring; **unverified** — follows the cited method as implemented by contributors; the equations have not been checked against the publication. Recovery of planted structure is tested, nothing more. ⚡ requires the `deep` extra.
<!-- algorithms:end -->

Each estimator's docstring gives its objective and update equations,
parameters, attributes, references and any deviation from the cited method.
The [documentation](https://soft-clustering.readthedocs.io/en/latest/) renders
them as one page per algorithm.

### Choosing an algorithm

- **Data you would give to k-means:** FCM is the baseline. GK and Gath–Geva
  allow ellipsoidal clusters; GMM gives calibrated probabilities; PCM and PFCM
  are robust to outliers; ECM additionally flags ambiguous samples (mass on
  sets of clusters) and outliers (mass on the empty set).
- **Non-convex clusters:** AFCM (graph embedding) or SoftKSC (kernel spectral).
- **Unknown number of clusters:** SCM, SoftDBSCANGM, CAFHFCM, and for
  documents WBSC, SISC and KMART.
- **Graphs:** BayesianNMF and MMSB for mixed memberships, BIGCLAM and NOCD for
  overlapping communities, CDCGS and DMoN for modularity-based soft partitions.
- **Text:** LDA or PLSI for topic proportions.
- **Several clusterings to reconcile:** SMCLA (soft consensus), SCSPA, SHBGF.
- **Images:** AFCMAdaptive (intensity inhomogeneity), SKFCM (spatial
  smoothness), RDFKC (learned features).

## Reproducibility

- Pass an integer `random_state` to get identical results across runs and
  refits of the same instance. Seeded estimators never read or reseed NumPy's
  global generator, and the deep estimators run in a forked PyTorch random
  stream. `tests/test_protocol.py` checks all three properties for every
  estimator.
- `random_state=None` is not reproducible. For the few estimators written
  against NumPy's legacy global generator (KFCM, MBMM, BayesianNMF), `None`
  draws from that global generator, as scikit-learn does.
- Results can still differ across platforms in the last floating-point bits
  (BLAS reduction order), which can change the outcome of a tie.

## Benchmarks

The `soft_clustering.benchmarking` subpackage (install the `bench` extra)
measures runtime, memory, scalability and clustering quality, and includes 20
datasets and 12 validity indices:

```python
from soft_clustering import FCM, GK, PCM
from soft_clustering.benchmarking import (
    ClusteringBenchmark, ClusteringQualityBenchmark, RuntimeBenchmark, get_dataset,
)

X, y = get_dataset("iris")
results = ClusteringBenchmark(
    models=[FCM(n_clusters=3, random_state=0),
            GK(n_clusters=3, random_state=0),
            PCM(n_clusters=3, random_state=0)],
    benchmarks=[RuntimeBenchmark(n_repeats=3), ClusteringQualityBenchmark()],
).run(X, y)
```

The cross-family comparison and the agreement study against scikit-fuzzy and
scikit-learn are regenerated by the scripts in [`benchmarks/`](benchmarks/);
see [`benchmarks/README.md`](benchmarks/README.md). The committed results
were generated with the corrected 0.2.0 code; each comes with a provenance
file recording the platform, library versions, seeds and SCPP commit.

<a id="testing"></a>
## Testing

```bash
git clone https://github.com/soft-clustering/soft-clustering.git
cd soft-clustering
pip install -e ".[dev,deep]"
pytest
```

The suite is layered by what a passing test establishes:

| Layer | File | Establishes |
| --- | --- | --- |
| Conformance | `tests/test_protocol.py` | every estimator honours the protocol, is reproducible, leaves global random state alone, refits identically, and does not modify its input |
| Recovery | `tests/test_recovery.py` | every estimator recovers planted structure in an easy instance of its input modality |
| External agreement | `tests/test_external_agreement.py` | agreement with scikit-fuzzy, scikit-learn and evclust (the *validated* status) |
| Per estimator | `tests/test_<name>.py` | the method's equations, invariants (e.g. monotone objectives) and regression tests |
| Edge cases | `tests/test_edge_cases.py` | duplicates, singular covariances, extreme fuzzifiers, non-finite input: a valid result or a clear error |
| Registry | `tests/test_registry.py` | the algorithm table matches the code |

Without the `deep` extra the PyTorch tests are skipped, not errored.

## Getting help

- **Questions about usage:** open an issue with the
  [question template](https://github.com/soft-clustering/soft-clustering/issues/new/choose).
- **Bugs:** open an issue with the bug-report template; include the SCPP
  version (`soft_clustering.__version__`), a minimal example and the full
  traceback.
- **A method you would like added:** use the algorithm-request template.

## Contributing

Contributions, bug reports and algorithm proposals are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the estimator protocol and the checklist
for adding an algorithm, and the [code of conduct](CODE_OF_CONDUCT.md). Please
report security issues privately as described in
[SECURITY.md](.github/SECURITY.md).

## Citation

If you use SCPP, please cite the software (GitHub's *Cite this repository*
button reads [`CITATION.cff`](CITATION.cff)) and the accompanying preprint:

```bibtex
@misc{rezaee2026scpp,
  title         = {{SCPP}: A Unified Python Library for Soft Clustering},
  author        = {Kiyan Rezaee and Morteza Ziabakhsh and Artin Bahrampour and
                   Seyed Mohammad Ghoreishi and Asal Khajeh and Ali Sajedifar and
                   Manny Chalak and Ava Zerafatangiz and Sadegh Eskandari},
  year          = {2026},
  eprint        = {2607.19620},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.19620},
}
```

Please also cite the original publication of each algorithm you use; the
table above and every estimator's docstring give it.

## License

[MIT](LICENSE).
