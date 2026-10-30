User guide
==========

Who SCPP is for
---------------

SCPP is for researchers and practitioners who want to apply or compare soft
clustering methods without re-implementing them: data scientists choosing a
method for overlapping or uncertain cluster structure, researchers who need a
baseline from the literature, and method developers who want to benchmark a
new algorithm against established ones under identical conditions.

Installation
------------

.. code-block:: bash

   pip install soft-clustering

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Extra
     - Command
     - Needed for
   * - *(none)*
     - ``pip install soft-clustering``
     - every estimator except the four deep ones
   * - ``deep``
     - ``pip install "soft-clustering[deep]"``
     - CDCGS, DMoN, NOCD, RDFKC
   * - ``bench``
     - ``pip install "soft-clustering[bench]"``
     - the :doc:`benchmarking suite <benchmarking>`
   * - ``baselines``
     - ``pip install "soft-clustering[baselines]"``
     - agreement tests against scikit-fuzzy, evclust and networkx
   * - ``dev``
     - ``pip install -e ".[dev]"``
     - running the test suite and the linters

SCPP supports Python 3.10 to 3.14.

The estimator protocol
----------------------

Soft clustering algorithms differ in what they consume: prototype methods take
a feature matrix, graph methods an adjacency matrix, topic models documents,
consensus methods a list of membership matrices. SCPP does not hide that: the
argument of ``fit`` is whatever the method needs. What is shared is everything
after ``fit``:

``memberships_``
    ``(n_samples, n_clusters_)`` non-negative soft output. What it *means*
    depends on the method — see `What a membership means`_.
``labels_``
    The arg-max of ``memberships_``, or ``-1`` for a sample with no membership
    in any cluster (possible only for methods whose rows need not sum to one).
``centers_``
    ``(n_clusters_, n_features)`` prototypes for prototype-based methods,
    ``None`` otherwise.
``n_clusters_``
    The number of clusters actually produced. Methods that discover the count
    (SCM, SoftDBSCANGM, CAFHFCM, WBSC, SISC, KMART) report it here.

Methods:

- ``fit(...)`` returns the estimator; ``fit_predict(...)`` returns
  ``memberships_``.
- ``predict()`` / ``predict_proba()`` with no argument return the fitted
  partition. Given new data, **inductive** estimators apply their membership
  rule to it: FCM, GK, PCM, PFCM, GMM, Gath–Geva, ECM, ENTROPYFCM, KFCM, KFCCL,
  FCC, AFCMSimple, SCM and SoftKSC. The others are transductive and raise
  ``NotImplementedError``.
- ``get_params`` / ``set_params`` support parameter introspection and
  ``sklearn.base.clone``; SCPP's setter invalidates fitted state as described
  under "Relation to scikit-learn" below.

The number of clusters is always the constructor argument ``n_clusters``.
Spellings used by earlier releases (``c``, ``k``, ``K``, ``n_topics``,
``n_components``, ``n_communities``, ``n_blocks``) are still accepted.

What a membership means
-----------------------

The same attribute holds different quantities for different methods, and they
are not interchangeable in meaning:

- **Fuzzy memberships** (FCM and its relatives) are degrees of sharing that sum
  to one; they are not probabilities.
- **Posterior probabilities** (GMM, MBMM, BGMM) are calibrated under the
  model's distributional assumptions.
- **Typicalities** (PCM) measure how typical a sample is of each cluster
  independently; an outlier is atypical of all of them and its row sums to
  much less than one.
- **Evidential** methods (ECM, EVCLUS) produce a *credal partition* in
  ``masses_`` — mass on single clusters, on sets of clusters (ambiguity) and,
  for ECM, on the empty set (outliers). ``memberships_`` is its pignistic
  transform.
- **Affiliations** (BIGCLAM, NOCD) are unnormalised strengths; a node can be
  strongly affiliated to several communities.
- **Topic proportions** (LDA, PLSI) and **mixed memberships** (MMSB,
  BayesianNMF) are distributions over latent components.
- **Degrees of match** (WBSC, SISC, KMART) are binary: a document matches
  several clusters, one, or none.
- **Crisp consensus** (SCSPA, SHBGF) is one-hot; SMCLA's consensus is soft.

Each :doc:`algorithm page <algorithms/index>` states which applies.

Choosing an algorithm
---------------------

- *Feature vectors, clusters like k-means:* FCM is the baseline. GK and
  Gath–Geva fit ellipsoidal clusters; GMM gives probabilities; PCM and PFCM
  are robust to outliers; ECM flags ambiguous samples and outliers explicitly.
- *Non-convex clusters:* AFCM or SoftKSC.
- *Unknown number of clusters:* SCM, SoftDBSCANGM or CAFHFCM.
- *Graphs:* BayesianNMF or MMSB for mixed memberships; BIGCLAM or NOCD for
  overlapping communities; CDCGS or DMoN for modularity-based soft partitions.
- *Text:* LDA or PLSI.
- *Several clusterings to reconcile:* SMCLA, SCSPA or SHBGF.
- *Images:* AFCMAdaptive for intensity inhomogeneity, SKFCM for spatial
  smoothness, RDFKC for learned features.

Prefer estimators with status *validated* or *reference* when the result will
be reported; treat *unverified* ones as starting points.

Verification status
-------------------

Every estimator carries one of four levels, also returned by
:func:`soft_clustering.list_algorithms`:

- **validated** — agrees with an independent implementation (scikit-fuzzy,
  scikit-learn, evclust), or reduces exactly to one in a limit.
- **reference** — implements the update equations of the cited publication,
  checked against it; invariants such as monotone objectives and recovery of
  planted structure are tested.
- **adapted** — implements the cited method with deviations documented in the
  estimator's docstring.
- **unverified** — the equations have not been checked against the
  publication; only recovery of planted structure is tested.

Every estimator, whatever its status, is tested to recover planted structure
in an easy instance of its input modality.

Reproducibility
---------------

- Every stochastic estimator accepts ``random_state``. An integer makes a fit
  reproducible, across runs and across refits of the same instance.
- A seeded fit neither reads nor reseeds NumPy's global random generator, and
  the PyTorch estimators run in a private, forked random stream, so fitting a
  model never changes the random numbers the rest of your program sees.
- ``random_state=None`` is not reproducible. KFCM, MBMM and BayesianNMF then
  draw from NumPy's global generator, as scikit-learn estimators do.
- Results may differ across platforms in the last floating-point bits.

Relation to scikit-learn
------------------------

Every estimator implements ``get_params`` and ``set_params``, declares
scikit-learn estimator tags, raises scikit-learn's ``NotFittedError`` (which
is also a ``RuntimeError``), and works with ``sklearn.base.clone``.

SCPP deliberately reconstructs constructor state on every successful nonempty
``set_params`` call, even for unchanged values. This discards the old fit and
other instance additions; refit before predicting. Constructor parameter
objects are passed through without cloning. An empty call is a no-op, and
unknown keys or constructor rejection preserve the existing instance. Checks
performed only during ``fit`` remain deferred until fitting. The setter accepts
the existing ``get_params`` keys, without nested parameter expansion. This
reconstruction policy differs from scikit-learn's general setter behavior.

The inductive estimators that take a feature matrix — FCM, GK, GMM, PCM,
PFCM, Gath–Geva, ECM, ENTROPYFCM, AFCMSimple, KFCM and FCC — pass
scikit-learn's ``check_estimator`` suite, except two sparse-input checks
that assume ``predict_proba`` belongs to a classifier. ``tests/test_sklearn_compat.py``
runs the suite. They accept and ignore ``y`` in ``fit``, so they work inside
pipelines.

Deliberate differences:

- ``fit_predict`` returns memberships, not labels; use ``fit(X).labels_``.
- Estimators for graphs, documents, images or several inputs take that input
  in ``fit``; scikit-learn tools that assume ``fit(X, y)`` do not apply to
  them.
- ``predict(X)`` raises ``NotImplementedError`` for transductive estimators.
- No estimator defines ``score``, so model selection needs an explicit
  ``scoring`` argument.

Input validation
----------------

Feature matrices are converted to float64 and checked before fitting. NaN or
infinite values, a wrong number of dimensions, too few samples for the
requested clusters and out-of-range hyperparameters raise ``ValueError`` (or
``TypeError`` for a wrong type) with a message naming the estimator and the
problem. Inputs are never modified in place.

Most estimators also check the types of their arguments at call time with
`typeguard <https://typeguard.readthedocs.io/>`_, so a wrong type fails at
the call rather than deep inside the algorithm. The check covers arguments
and return values of public methods, not the inner loops. Measured on fits
of FCM, GK, PFCM, ENTROPYFCM, KFCM, GMM and rough k-means (200 and 5000
samples), its cost was within timing noise. Running Python with ``-O``
disables it.
