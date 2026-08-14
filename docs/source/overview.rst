SCPP: Soft Clustering Python Package - Documentation
======================================================


.. image:: https://raw.githubusercontent.com/soft-clustering/soft-clustering/refs/heads/main/SCPP_Poster.png


SCPP collects 42 soft clustering algorithms behind one estimator protocol, spanning fuzzy, possibilistic, evidential, probabilistic, graph, document, ensemble and deep methods, together with a benchmarking suite for comparing them on equal terms.

This package is designed to support research and applied workflows involving clustering under uncertainty, overlapping memberships, and soft assignments.


Highlights
----------

- **One protocol, 42 algorithms** — Every estimator exposes ``memberships_``, ``labels_``, ``centers_`` and ``n_clusters`` after ``fit``, regardless of whether the method consumes a feature matrix, a graph, raw documents, or an ensemble of partitions.
- **Checked, not asserted** — A conformance suite fits all 42 estimators on every commit and verifies the contract: membership shape, the partition constraint where the formulation imposes one, out-of-sample behaviour, ``sklearn.clone`` round-tripping, and reproducibility under a fixed ``random_state``.
- **Benchmarking included** — Runtime, memory, scalability and clustering quality, over 20 datasets and 12 validity metrics — shipped *inside* the package.
- **Light by default** — A base install requires only ``numpy``, ``scipy``, ``scikit-learn`` and ``typeguard`` — enough to fit 38 of the 42 estimators. PyTorch and pandas live behind extras you opt into.
- **Typed** — Ships ``py.typed``; hints are checked at runtime by ``typeguard`` and visible to your type checker.


Installation
------------

The package can be installed from PyPI:

.. code-block:: bash

   pip install soft-clustering

Optional extras, separated by what they are needed **for**:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Extra
     - Install
     - Needed for
   * - *(base)*
     - ``pip install soft-clustering``
     - Fitting any of the 38 non-deep estimators
   * - ``deep``
     - ``pip install "soft-clustering[deep]"``
     - CDCGS, DMoN, NOCD, RDFKC
   * - ``bench``
     - ``pip install "soft-clustering[bench]"``
     - Running ``soft_clustering.benchmarking``
   * - ``baselines``
     - ``pip install "soft-clustering[baselines]"``
     - Agreement checks against third-party implementations
   * - ``docs``
     - ``pip install "soft-clustering[docs]"``
     - Building the documentation
   * - ``dev``
     - ``pip install -e ".[dev,deep]"``
     - Developing and testing

Requires Python 3.10 or newer.


Quick Start
-----------

Basic usage with the package API:

.. code-block:: python

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

Every estimator follows the same shape, so swapping the method is a one-word change:

.. code-block:: python

   from soft_clustering import ECM, GK, PCM, RoughKMeans

   for cls in (GK, PCM, ECM, RoughKMeans):
       model = cls(n_clusters=2).fit(X)
       print(f"{cls.__name__:20s} {model.memberships_.shape}")


Algorithm Catalogue
--------------------

Grouped by what the algorithm consumes. ⚡ marks estimators requiring the ``deep`` extra.

**Feature matrix** — 26 estimators

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Estimator
     - Full Name
     - Description
   * - :doc:`FCM <algorithms/fcm>`
     - Fuzzy C-Means
     - The canonical membership-based method
   * - :doc:`PCM <algorithms/pcm>`
     - Possibilistic C-Means
     - Typicalities; rows need not sum to 1
   * - :doc:`PFCM <algorithms/pfcm>`
     - Possibilistic Fuzzy C-Means
     - Combines memberships and typicalities
   * - :doc:`GK <algorithms/gk>`
     - Gustafson–Kessel
     - Adaptive per-cluster covariance
   * - :doc:`ECM <algorithms/ecm>`
     - Evidential C-Means
     - Belief masses over sets of clusters
   * - :doc:`KFCM <algorithms/kfcm>`
     - Kernelized Fuzzy C-Means
     - Kernel-space distances
   * - :doc:`SKFCM <algorithms/skfcm>`
     - Spatially-Constrained Kernelized FCM
     - Adds an image neighbourhood term
   * - :doc:`KFCCL <algorithms/kfccl>`
     - Kernel-based Fuzzy Competitive Learning
     - Competitive-learning update
   * - :doc:`CAFCM <algorithms/cafcm>`
     - Collaborative Annealing FCM
     - Deterministic annealing
   * - :doc:`CAFHFCM <algorithms/cafhfcm>`
     - Centroid Auto-Fused Hierarchical FCM
     - Fuses centroids hierarchically
   * - :doc:`ENTROPYFCM <algorithms/EntropyFCM>`
     - Entropy c-Means
     - Entropy regularisation in place of the fuzzifier
   * - :doc:`AFCM <algorithms/afcm>`
     - AFCM with full graph embedding
     - Graph-regularised memberships
   * - :doc:`AFCMSimple <algorithms/afcmSimple>`
     - AFCM without graph embedding
     - The unregularised variant
   * - :doc:`AFCMAdaptive <algorithms/afcmadaptive>`
     - Adaptive FCM for image segmentation
     - Operates on a single image
   * - :doc:`FCC <algorithms/fcc>`
     - Fuzzy Color Clustering
     - Fuzzy colour spheres in CIELAB space
   * - :doc:`RPFKM <algorithms/rpfkm>`
     - Robust Projected Fuzzy K-Means
     - Joint dimensionality reduction for noisy, high-dimensional data
   * - :doc:`SFCMEP <algorithms/sfcmep>`
     - Semi-supervised Fuzzy Clustering with Membership Prior
     - Consumes a partially labelled target vector
   * - :doc:`FeMIFuzzy <algorithms/femifuzzy>`
     - Federated Multiple Imputation Fuzzy Clustering
     - Consumes per-client matrices
   * - :doc:`RoughKMeans <algorithms/rough_k_means>`
     - Rough K-Means
     - Lower and upper approximations
   * - :doc:`SCM <algorithms/scm>`
     - Subtractive Clustering Method
     - Determines the cluster count itself
   * - :doc:`SoftDBSCANGM <algorithms/softdbscangm>`
     - Soft DBSCAN with Gaussian mixtures
     - Density-based, discovers *k*
   * - :doc:`SoftKSC <algorithms/softksc>`
     - Soft Kernel Spectral Clustering
     - Semi-supervised; two non-parallel hyperplanes
   * - :doc:`GMM <algorithms/gmm>`
     - Gaussian Mixture (EM)
     - Responsibilities as memberships
   * - :doc:`BGMM <algorithms/bgmm>`
     - Beta-Gaussian Mixture Model
     - Two aligned views
   * - :doc:`MBMM <algorithms/mbmm>`
     - Multivariate Beta Mixture Model
     - For data on the unit interval
   * - :doc:`RDFKC <algorithms/rdfkc>` ⚡
     - Robust Deep Fuzzy K-Means
     - Image tensors

**Graphs and networks** — 6 estimators

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Estimator
     - Full Name
     - Description
   * - :doc:`BIGCLAM <algorithms/bigclam>`
     - Cluster Affiliation Model for Big Networks
     - Overlapping communities at scale
   * - :doc:`BayesianNMF <algorithms/bnmf>`
     - Bayesian NMF for overlapping communities
     - Automatic relevance determination
   * - :doc:`MMSB <algorithms/mmsb>` ⚡
     - Mixed Membership Stochastic Blockmodel
     - Generative blockmodel
   * - :doc:`NOCD <algorithms/nocd>` ⚡
     - Neural Overlapping Community Detection
     - GNN + Bernoulli–Poisson
   * - :doc:`DMoN <algorithms/dmon>` ⚡
     - Deep Modularity Networks
     - Modularity-optimising pooling
   * - :doc:`CDCGS <algorithms/cdcgs>` ⚡
     - Community Detection via Gumbel Softmax
     - Differentiable assignment

**Documents and text** — 5 estimators

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Estimator
     - Full Name
     - Description
   * - :doc:`LDA <algorithms/lda>`
     - Latent Dirichlet Allocation
     - Topic–word and document–topic factors
   * - :doc:`PLSI <algorithms/plsi>`
     - Probabilistic Latent Semantic Indexing
     - Likelihood-based topic model
   * - :doc:`SISC <algorithms/sisc>`
     - Similarity-Based Soft Clustering
     - Discovers the cluster count
   * - :doc:`KMART <algorithms/kmart>`
     - Modified Fuzzy ART for documents
     - Adaptive resonance
   * - :doc:`WBSC <algorithms/wbsc>`
     - Word-Based Soft Clustering
     - Word-driven soft assignment

**Ensembles and consensus** — 3 estimators

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Estimator
     - Full Name
     - Description
   * - :doc:`SCSPA <algorithms/scspa>`
     - Soft CSPA
     - Similarity-based consensus over soft partitions
   * - :doc:`SHBGF <algorithms/shbgf>`
     - Soft HBGF
     - Bipartite graph consensus over concatenated memberships
   * - :doc:`SMCLA <algorithms/smcla>`
     - Soft MCLA
     - Groups clusters into meta-clusters


Benchmarking
------------

The benchmarking suite ships inside the package, so it is available straight
after installation.

.. code-block:: bash

   pip install "soft-clustering[bench]"

.. code-block:: python

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

**Included backends:**

- ⏱️ ``RuntimeBenchmark`` — Fit and predict time, with repeats and standard deviation
- 💾 ``MemoryBenchmark`` — Resident set size sampled *during* the fit
- 📈 ``ScalabilityBenchmark`` — Runtime and memory as the sample count grows
- 🎯 ``ClusteringQualityBenchmark`` — Silhouette, Calinski–Harabasz, Davies–Bouldin; ARI and NMI when labels are given; partition coefficient and entropy

Also included: 20 datasets (bundled, synthetic and OpenML) and 12 validity metrics.

See :doc:`benchmarking` for the full reference.


Testing
-------

The project includes a comprehensive test suite in the `tests/`_ directory, covering all implemented algorithms.

To run the tests:

.. code-block:: bash

   pip install -e ".[dev,deep]"
   pytest

See `tests/HOW_TO_RUN.txt`_ for more details.

.. _`tests/`: https://github.com/soft-clustering/soft-clustering/tree/main/tests
.. _`tests/HOW_TO_RUN.txt`: https://github.com/soft-clustering/soft-clustering/blob/main/tests/HOW_TO_RUN.txt


Development
-----------

.. code-block:: bash

   git clone https://github.com/soft-clustering/soft-clustering.git
   cd soft-clustering
   pip install -e ".[dev,deep]"

   pytest                                                    # tests
   pytest --cov=soft_clustering --cov-report=term-missing    # with coverage
   ruff check soft_clustering tests example tools            # lint
   black --check soft_clustering tests example tools         # formatting
   sphinx-build -b html -W docs/source docs/_build/html      # docs, warnings are errors

See **CONTRIBUTING.md** for the estimator protocol and what adding an algorithm involves.


Citation
--------

This package accompanies a paper submitted to the JMLR MLOSS track.

- **Paper:** `arXiv:2607.19620 <https://arxiv.org/abs/2607.19620>`_

.. code-block:: text

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


Contributing
------------

Contributions, bug reports and algorithm proposals are all welcome.

- `Contributing guidelines <https://github.com/soft-clustering/soft-clustering/blob/main/CONTRIBUTING.md>`_
- `Code of conduct <https://github.com/soft-clustering/soft-clustering/blob/main/CODE_OF_CONDUCT.md>`_
- `Security policy <https://github.com/soft-clustering/soft-clustering/blob/main/.github/SECURITY.md>`_ — please report vulnerabilities privately
- `Open an issue <https://github.com/soft-clustering/soft-clustering/issues/new/choose>`_


License
-------

Distributed under the terms of the `MIT license <https://github.com/soft-clustering/soft-clustering/blob/main/LICENSE>`_.


API Reference
-------------

.. toctree::
  :maxdepth: 2

  benchmarking
  algorithms/index
