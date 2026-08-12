# Changelog

All notable changes to SCPP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — unreleased

This release makes the library's central claim true: every exported estimator
now fits, exposes the documented soft output, and is checked against the
protocol in CI. Several algorithms did not previously do any of those things.
It contains breaking changes, all of them corrections.

### Fixed — algorithms

- **MMSB** performed no inference. `get_memberships()` returned samples from
  the Dirichlet *prior*, drawn in the constructor and never conditioned on a
  graph; there was no `fit`. Replaced with the naive mean-field variational EM
  of Airoldi et al. (2008). A uniform start sits at a symmetric fixed point
  where every node is equally mixed, so the variational factors are seeded
  from a spectral partition (`init="spectral"`); planted blockmodels are now
  recovered at ARI 1.0. Because the inference is pure NumPy/SciPy, MMSB has
  left the `deep` extra.
- **CDCGS** and **DMoN** were `nn.Module`s with `forward` and `loss` but no
  training procedure, no `fit`, and no membership output. Both now implement
  the estimator protocol. CDCGS optimises the soft modularity its reference
  targets, with Gumbel temperature annealing and `n_init` restarts (a single
  run reaches modularity 0.30 against a planted partition's 0.46; five
  restarts reach 0.46). DMoN's collapse regulariser now follows Equation (7)
  of Tsitsulin et al. (2023). `forward` and `loss` keep their previous
  signatures.
- **FeMIFuzzy** had four defects. `fit_predict` returned only the global
  centroids and computed the memberships into a discarded expression, so the
  estimator produced no soft output; the fuzzy c-means membership update
  summed the same term `C` times instead of over clusters and wrote
  `2 / self.m - 1` where `2 / (m - 1)` was meant; Xie–Beni model selection
  swept from `C = 1`, where the index is undefined and evaluates to zero, so
  every fit collapsed to a single cluster; and the Sammon mapping used an
  unseeded RNG with a step size roughly four orders of magnitude too small, so
  it returned its random initialisation and destroyed the data's structure.
  The projection is now a deterministic, PCA-initialised diagonal-Newton
  Sammon mapping (distance correlation 0.9998), and the estimator recovers
  planted structure.
- **SoftDBSCANGM**, **PFCM** and **RPFKM** declared
  `_partition_constrained = False` while normalising memberships
  unconditionally. That silently disabled the conformance check and made the
  fuzziness indices report `nan` for them.

### Added — algorithms

- **`GathGeva`** — Gath–Geva fuzzy maximum-likelihood estimation clustering
  (1989). Listed in the paper's algorithm table but never implemented. The
  exponential distance is evaluated in log space, so it does not overflow, and
  it is seeded from fuzzy c-means as the source prescribes.
- **`EVCLUS`** — evidential clustering of proximity data (Denœux and Masson,
  2004). Also listed but never implemented. Exposes the credal partition as
  `masses_` and the pignistic transform as `memberships_`; the analytic stress
  gradient is checked against finite differences.

The library now exports **42** estimators.

### Fixed — metrics

- **`fuzzy_hypervolume` computed the wrong quantity.** It returned
  `mean(prod(U, axis=1))`, which takes neither the data nor the prototypes,
  measures fuzziness rather than volume, and underflows to `1e-27` at
  `K = 20`. Replaced with the Gath–Geva index `sum_i sqrt(det F_i)` over the
  fuzzy covariances, verified against the analytic volume of isotropic
  clusters. **The signature changed** to `fuzzy_hypervolume(X, U, centers, m)`.
- The partition coefficient, modified partition coefficient and partition
  entropy are only defined when the rows of `U` sum to one. They now return
  `nan` for unnormalised memberships instead of a number on a different
  scale. `soft_clustering_metrics` detects this from `U`, or takes an explicit
  `partition_constrained` argument.
- `clustering_metrics` and `soft_clustering_metrics` always return every key,
  with `nan` where a metric does not apply, instead of omitting keys and
  producing ragged benchmark rows.
- `xie_beni_index` returns `inf` when all prototypes coincide. It previously
  divided by an infinite separation and reported a perfect `0.0` for a fully
  degenerate solution.
- `ClusteringQualityBenchmark` now calls `metrics.py` instead of
  reimplementing the metrics, so a benchmark row and a direct metric call
  cannot disagree.

### Changed — API (breaking)

- **`predict(X_new)` no longer silently returns the training labels.** Most
  soft clustering algorithms are transductive; returning `labels_` for unseen
  data hands back a partition of different data, of the wrong length, with no
  error. Transductive estimators now raise `NotImplementedError`.
  `predict()` with no argument is unchanged. Estimators with a genuine
  out-of-sample rule set `_supports_out_of_sample`; `SoftKSC` is the only one.
- `predict` previously had three incompatible signatures across the library
  (`predict(self)`, `predict(self, X)`, `predict(self, X=None)`), so
  `model.predict(X)` was not uniformly callable. All estimators now use
  `predict(self, X=None)`.
- **`SoftKSC.predict` returns cluster indices `{0, 1}`**, matching the
  library-wide rule `labels_ == argmax(U)`. The signed `{-1, +1}` encoding
  moved to `SoftKSC.signed_labels()`.
- **`FeMIFuzzy.fit_predict` returns the global membership matrix**, not a list
  of centroids. The centroids remain available as `centers_`.
- **`AFCMAdaptive.predict` returns the flat `(H*W,)` labelling** required by
  the protocol. `label_map()` returns the `(H, W)` image-shaped assignment.
- `BaseSoftClusterer` gains `get_params` / `set_params`, so `sklearn.clone`
  and scikit-learn parameter introspection work. Estimators are still not
  drop-in scikit-learn clusterers: `fit` signatures differ by input modality
  and none defines `score`.
- `LDA` publishes `doc_topic_` and `memberships_`.
- `scikit-learn` moved from the `bench` extra to a core dependency. Eight
  estimators import it at module level, so six of them were unimportable on
  the documented base install.

### Added — testing and infrastructure

- The conformance suite covers **all 42 estimators**. It previously excluded
  ten behind a guard that permitted excluding a third of the library, while
  the documentation claimed it fitted every exported estimator. Exclusions are
  now a hard failure.
- New conformance checks: `_partition_constrained` declarations must not
  understate the guarantee; `predict(X_new)` must either work or raise;
  `get_params` must round-trip through `sklearn.clone`.
- `benchmarks/run_main_benchmark.py` — the cross-family comparison over the
  shipped 20-dataset suite, with the results consolidated into the paper's
  tables by one command.
- `benchmarks/run_external_baselines.py` and `tests/test_external_agreement.py`
  — agreement and runtime against `scikit-fuzzy` and `scikit-learn`.
- Numerical edge-case tests: duplicate points, `K > n`, single cluster,
  extreme fuzzifier, singular covariance, empty graphs, `float32` input.
- CI runs the full three-OS by four-Python matrix. It previously ran macOS and
  Windows on one interpreter each while the documentation described the full
  cross-product.
- The `minimal-install` CI job walks the estimator registry and fits one
  estimator per input modality. It previously ran `import soft_clustering`,
  which — because estimators are imported lazily — touched none of them and
  passed throughout the period when six were unimportable.
- `tools/paper_stats.py` runs in CI, so the reported counts cannot drift from
  the source again.

## [0.0.3] — 2026-07-23

Initial PyPI releases (0.0.1, 0.0.2, 0.0.3, all on the same day).

Known defects, all fixed in 0.2.0: the published wheel imported every
estimator eagerly, so `import soft_clustering` failed without `scikit-learn`
and `torch` even though neither was declared; `myst_parser`, a Sphinx plugin,
was declared as a runtime dependency; and the `benchmarking` subpackage was
absent from the distribution.
