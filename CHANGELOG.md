# Changelog

All notable changes to SCPP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — unreleased

This release makes the library's central claim true: every exported estimator
now fits, exposes the documented soft output, recovers planted structure, and
is checked against the protocol in CI. A correctness audit found that a third
of the estimators did not compute what their reference describes. It contains
breaking changes, all of them corrections.

### Correctness audit

Every estimator was read against its cited publication (or the authors' own
later description) and tested for recovery of planted structure. The
conformance suite had passed throughout; it checks the *form* of the output,
and several estimators produced well-formed output that was wrong. Each fix
below has a regression test.

#### Did not implement the cited method

- **ECM** assigned mass only to the singletons and the empty set, which is
  noise clustering, not evidential c-means. Reimplemented with the full set
  of focal sets (or `focal_sets="simple"`), barycentric focal-set prototypes,
  the `H V = B` prototype system and the pignistic transform of Masson and
  Denoeux (2008). From identical initial prototypes it now agrees with
  `evclust` (the authors' package) to 1e-6 in prototypes and masses.
  `masses_` has one column per focal set plus the empty set (column 0).
- **ENTROPYFCM** ignored `entropy_weight` and computed plain fuzzy c-means.
  It now implements entropy-regularised (maximum-entropy) c-means, Li and
  Mukaidono (1995); the fuzzifier `m` has been removed. The paper cites Gupta
  et al. (2021), an evolutionary multi-objective method that is not
  implemented.
- **SoftKSC** was a two-class semi-supervised kernel ridge classifier,
  `fit(X_labeled, y_labeled, X_unlabeled)`. It now implements soft kernel
  spectral clustering (Langone, Mall and Suykens, 2013): the KSC dual
  eigenproblem, ECOC encoding, cosine-distance prototypes and soft
  memberships, with an out-of-sample rule. `fit(X)`, `n_clusters`, `gamma`.
- **AFCM** and **AFCMSimple** applied the fuzzy c-means ratio rule with a
  fixed fuzzifier. They now implement the adaptive model of Chen et al.
  (2024): entropy-type memberships with a learned precision, and for AFCM a
  spectral embedding regularised by a normalised Gaussian kNN Laplacian. AFCM
  now separates non-convex clusters (two moons, concentric circles) that FCM
  cannot. `m` has been removed from both.
- **AFCMAdaptive** computed class intensities as `sum(u y) / sum(u y^2)`
  (roughly the reciprocal of the intensity) and updated the gain field with a
  Gaussian blur; it segmented a noise-free two-level image with an adjusted
  Rand index of 0. Reimplemented as in Pham and Prince (1999), with the gain
  field solved from its sparse positive-definite system.
- **SKFCM** used a kernel-space centroid that needs the dense
  `(H*W) x (H*W)` kernel (34 GB for a 256 x 256 image). Reimplemented as in
  Zhang and Chen (2004), in `O(H W K)` memory.
- **CAFHFCM** applied a squared-distance shrinkage step to a fixed number of
  centroids. It now minimises the fused objective of Lin and Chen (2021),
  fuzzifier 2 plus an unsquared L2 fusion penalty, by
  majorisation-minimisation, and merges fused centroids, so the number of
  clusters is discovered. Its parameter is the dimensionless `alpha`;
  `n_clusters` and `m` are no longer used. The default `alpha="auto"` traces
  the warm-started fusion path and returns the cluster count that is stable
  over the widest range of strengths, as the reference selects it; the path
  is exposed as `path_`.
- **CAFCM** ran a single annealing FCM. It now runs a population of
  collaborating modules re-initialised towards the best one, as Li and Wang
  (2024) describe; the specific meta-heuristic rule is SCPP's own and
  documented as such.
- **FCC** used `1 / (1 + d - jnd)` memberships and squared weights. It now
  follows Kim and Lee (2024), Eqs. 7, 17 and 18.
- **KFCCL** never separated its clusters: every membership was exactly 1/K
  on well-separated data. Reimplemented as kernel fuzzy competitive learning
  (prototypes as kernel expansions, online Gibbs-weighted updates). Its
  optimisation-equivalence tests were removed; see `optimization/README.md`.
- **SCM** implemented only Chiu's accept test and ignored `er`. The full
  accept / reject / gray-zone rule of Chiu (1994) is implemented.
- **FeMIFuzzy** Sammon-projected each client independently and averaged
  prototypes across those unaligned projections, matched global clusters by
  index, and failed on any missing value. The pipeline now imputes locally,
  clusters in the shared feature space, and aligns client prototypes by
  Hungarian matching before averaging.

#### Formula and logic errors

- **BayesianNMF**: the relevance-determination update was inverted. It is
  now `beta_k = (N + a - 1) / (0.5 (|w_k|^2 + |h_k|^2) + b)` (Psorakis et al.,
  2011, Eq. 19); relevance determination now prunes unneeded communities
  (`n_effective_`). Memberships are the row-normalised `W` (Eq. 20), and an
  `N x N` matrix of ones is no longer allocated.
- **BIGCLAM**: the gradient omitted the neighbour correction of the
  non-edge sum. Implemented with node-wise updates and line searches as in
  Yang and Leskovec (2013); the log-likelihood is non-decreasing.
- **PFCM** dropped the factor `b` from the typicality update, and
  re-estimated `gamma` every sweep. `gamma` is now computed once from an FCM
  partition and fixed, as Pal et al. (2005) prescribe.
- **PCM** re-estimated `eta` every sweep, so the iteration minimised no
  fixed objective. `eta` is computed once from an FCM partition
  (Krishnapuram and Keller, 1993) and the objective is monotone.
- **Gath-Geva** used `sum(u^m)/n` as the cluster prior, which is not a
  probability distribution; it is now `sum(u)/n`.
- **GMM** added `reg_covar` twice (M-step and E-step).
- **SFCMEP** re-weighted the previous sweep's memberships instead of the
  prior, so memberships collapsed to crisp assignments whatever `lam` was.
- **BGMM** started every component from identical parameters with uniform
  responsibilities, a fixed point of EM: every membership was exactly 1/K. It
  is now initialised from k-means.
- **RDFKC** kept its memberships in a torch tensor that the protocol did not
  read, so `memberships_` was the one-hot encoding of the labels. The
  membership step could also return negative values; it is now solved exactly
  on the simplex.
- **SMCLA** computed a soft consensus and returned only its arg-max.
- **PLSI**'s tempered EM lowered the temperature from the first iteration and
  stopped at the first dip of the held-out likelihood, usually after two
  iterations. Hofmann's schedule is implemented, `tempered=False` is the
  default (TEM targets generalisation and needs a large corpus), and
  `n_init` restarts were added.
- **SISC** chose the document *closest* to the existing centroids as the next
  centroid instead of the farthest, and derived its threshold from all
  centroid-document similarities, including self-similarities; most
  documents joined no cluster. Its refinement loop also had no iteration cap.
- **NOCD** computed its output in training mode (with dropout) and never
  restored the best weights it saved.
- **AFCM** initialisation is taken from the row-normalised spectral
  embedding (Ng, Jordan and Weiss, 2002).

#### State and side effects

- 18 estimators drew from NumPy's global random generator, and FCM, PCM, GK,
  GMM, PFCM, RDFKC and NOCD reseeded it in their constructors. NOCD also
  switched on `torch.use_deterministic_algorithms` for the whole process and
  changed the global default device; DMoN and CDCGS reseeded torch globally.
  Every stochastic estimator now takes `random_state`, a seeded fit never
  touches the global NumPy or torch state, and the torch estimators run in a
  forked random stream.
- KMART appended to the clusters of the previous fit; PLSI, SFCMEP and
  FeMIFuzzy created their random generators in the constructor; DMoN
  continued training from the previous fit's weights. Refitting an instance
  now reproduces the first fit.
- KFCM and KFCCL printed to standard output.

### Changed — API (breaking)

- `fit_predict` returns `memberships_` for every estimator. It returned a
  tuple, a dict, a sparse matrix or hard labels depending on the estimator.
- `n_clusters` is the constructor argument of every estimator with a given
  cluster count, including FCM, PCM, GK, GMM and NOCD, which took it at fit
  time. The old spellings (`c`, `k`, `K`, `n_topics`, `n_components`,
  `n_communities`, `n_blocks`) are still accepted as keywords; `fit(X, K)` is
  still accepted for FCM, PCM, GK, GMM and NOCD.
- `labels_` is `-1` for a sample with no membership in any cluster
  (possible only for estimators whose rows need not sum to one); it used to
  report cluster 0.
- `n_clusters_` is published after every fit and holds the number of
  clusters produced. **`fit` no longer overwrites the `n_clusters`
  parameter**; read the discovered count of SCM, SoftDBSCANGM, CAFHFCM, WBSC,
  SISC and KMART from `n_clusters_`.
- scikit-learn conventions: `get_params` reads parameters from the
  attributes of the same name (so `model.n_clusters = 4` is a parameter
  change); constructors no longer validate (KFCM and Gath-Geva raised there),
  `fit` does; unsupervised `fit` and `fit_predict` accept and ignore `y`;
  fitted attributes are not created in `__init__`; estimators define
  `__sklearn_tags__` and `__sklearn_is_fitted__`; using an unfitted estimator
  raises `soft_clustering.NotFittedError`, a subclass of both `RuntimeError`
  and scikit-learn's `NotFittedError`; and `n_features_in_` is set. The
  inductive feature-matrix estimators pass scikit-learn's `check_estimator`
  except two sparse-input checks (`tests/test_sklearn_compat.py`); before,
  FCM passed 11 of 41.
- Passing the cluster count to `fit`/`fit_predict` positionally (FCM, PCM,
  GK, GMM) emits a `DeprecationWarning`.
- RDFKC keeps the given `encoder`/`decoder` as parameters and trains copies
  (`encoder_`, `decoder_`); LDA keeps `alpha=None` and publishes the
  effective prior as `alpha_`; ECM's `prototypes`, PFCM's `gamma` and KFCM's
  `V`/`U` are read-only views of fitted attributes. DMoN's `hidden_channels`
  and `n_clusters` have defaults.
- `predict(X)` / `predict_proba(X)` work on new data for FCM, GK, PCM, PFCM,
  GMM, Gath-Geva, ECM, ENTROPYFCM, KFCM, KFCCL, FCC, AFCMSimple, SCM and
  SoftKSC.
- Feature matrices are validated: NaN and infinite values, wrong
  dimensionality and too few samples raise `ValueError` with a message naming
  the estimator; inputs are never modified.
- Removed parameters that had no role in the implemented method: `m` from
  ENTROPYFCM, AFCM and AFCMSimple; `C` from SoftKSC. Added `random_state`
  wherever there is randomness.
- Document estimators (KMART, SISC, WBSC) return dense memberships; SISC
  takes `n_clusters`, `max_iter` and `random_state`.
- `FuzzyCMeans`, `GustafsonKessel`, `GaussianMixtureEM` and
  `PossibilisticCMeans` are importable under their class names as well.
- The `deep` extra no longer installs `torch_geometric`: DMoN's single graph
  convolution is implemented in PyTorch (checked against `GCNConv`).

### Added

- `list_algorithms()` and `soft_clustering._registry`: one record per
  estimator (family, input, output semantics, how K is obtained, reference,
  verification status). The README table and the documentation pages are
  generated from it, and `tests/test_registry.py` binds it to the code.
- `tests/test_recovery.py`: every estimator must recover planted structure.
- Protocol checks for reproducibility, refit idempotence, untouched global
  random state, unmodified inputs and consistency of `predict_proba` on the
  training data.
- `tests/test_examples.py` runs every example script.
- The test suite runs without the `deep` extra (it previously failed at
  collection), and CI checks this in a dedicated job.
- `mypy` on the whole package in CI; `.pre-commit-config.yaml`. Bringing the
  algorithm modules under it corrected `RPFKM.fit_predict`'s return
  annotation and docstring (it returns the memberships, not a 3-tuple) and
  replaced the `None` placeholders that eight estimators assigned in
  `__init__` to attributes that only `fit` sets (`W`, `H`, `U`, `centers`,
  `B`, `pi`, `F`, ...); those attributes now exist only after fitting, as in
  scikit-learn.
- Benchmarks: every estimator is seeded, results are averaged over several
  seeds with provenance (commit, seeds, skipped datasets), kernel bandwidths
  use a documented scale heuristic, and an unreachable dataset is skipped
  rather than aborting the run.
- Benchmarks: FCC's `jnd` follows the data scale (a tenth of the RMS
  spread); its CIELAB default merged every point into one cluster on most
  datasets. `olivetti_faces` falls back to its OpenML copy (data id 41083)
  when scikit-learn's download is refused.
- `.zenodo.json`, so that a GitHub release can be archived with a DOI.
- Python 3.14 is supported and tested in CI (the suite, with and without
  the deep extra, passes on 3.10 to 3.14).

### Fixed — robustness

- **Stopping rules no longer depend on the units of the data.** FCM, PCM,
  GK, ECM, ENTROPYFCM and AFCMSimple stopped when the objective changed by
  less than an absolute `tol`; AFCM and BayesianNMF used
  `tol * max(|J|, 1)`, which is absolute whenever `|J| < 1`. Because these
  objectives scale with the squared data scale, the same data expressed in
  smaller units met the test after one or two sweeps (FCM on blobs scaled by
  1e-4 stopped after 2 sweeps instead of 6) and returned an unconverged
  partition. All eight now use a relative change of the objective. PFCM,
  RoughKMeans, CAFCM and GMM's k-means initialisation compared prototype
  movement with an absolute `tol`; it is now relative to the data spread
  (root mean per-feature standard deviation), as scikit-learn's `KMeans`
  does. Results on data of unit scale change only through the iteration at
  which a run stops. `tests/test_convergence.py` checks that FCM, PCM, PFCM,
  AFCMSimple, CAFCM and RoughKMeans return the same memberships and
  iteration count for data rescaled by 1e-4 and 1e4. Parameters that are in
  data units by definition (kernel bandwidths, DBSCAN's `eps`, ECM's
  `delta`, ENTROPYFCM's `entropy_weight`, the `reg_covar` ridge of GK,
  Gath-Geva and GMM, which follows scikit-learn) are unchanged.
- FCM's membership update added `1e-12` to every squared distance and PCM's
  typicality update floored squared distances at `1e-12`; both were
  redundant with the log-space evaluation and made memberships depend on the
  units of `X`. They are removed.
- KFCM's k-means++ seeding divided by zero when every sample coincided with
  an already chosen centre (for example, constant data), and could then
  leave a centre at the origin, which need not be a data point. It now draws
  uniformly from the same random number in that case.
- **WBSC was not reproducible across Python processes.** It built its
  initial word clusters by iterating over a set of strings, whose order
  changes with `PYTHONHASHSEED`, and its greedy merge depends on that order:
  the same corpus gave a differently ordered, and when merges occur possibly
  different, partition in every new interpreter. The vocabulary is now
  sorted. `tests/test_cross_process_reproducibility.py` fits all five
  document estimators under three hash seeds.
- **SFCMEP returned all-NaN memberships, without an error, when a class
  had a single labelled sample** (the usual semi-supervised setting). That
  sample is its class's initial centroid, so the prior's ratio
  `d_min / d_i` was `0 / 0`; the NaN spread through `rho_hat` to the whole
  prior. The ratio now takes its limit, one.
- SFCMEP replaced a zero prior entry by one, the largest prior weight. At
  `rho = 1` the non-labelled classes of every labelled sample have prior
  zero, so full trust in the labels enforced them less than `rho = 0.9`.
  Zero entries are now floored at the smallest positive double, and label
  enforcement is monotone in `rho` (tested).
- RPFKM formed `diff @ np.diag(s) @ diff.T`, allocating an `n x n` matrix
  per cluster in every iteration (0.8 GB at n = 10,000); it now computes the
  same scatter matrix as `(diff * s) @ diff.T` in `O(n d)` memory.

- `benchmarking.metrics.fuzzy_hypervolume` formed every cluster's d x d
  covariance at once and ran out of memory on high-dimensional data (5 GB
  for 40 clusters of 4096 features). It now works one cluster at a time,
  uses `slogdet`, and returns the exact value 0 when there are fewer samples
  than features. The benchmark worker records a scoring failure as
  `metric_error` instead of crashing.

- The documentation build failed where PyTorch is not installed (the docs CI
  job and Read the Docs): RDFKC's annotations were evaluated against the
  mocked module. They are now postponed, as in the other deep estimators.
- `myst-parser` is held below 4 in the docs requirements and the `docs`
  extra, because evclust (in the `dev` and `baselines` extras) requires it;
  the two sets of dependencies now install into one environment.

- `GathGeva(init_iter=0)` failed with a `TypeCheckError` from inside the
  initialisation; `init_iter` is now validated (`>= 1`).
- `EVCLUS` raised an `AttributeError` when the stress was not finite in any
  restart; it now raises a `RuntimeError` that names the likely cause.

### Performance

- SCM computes its potentials in row blocks and no longer stores the
  `(n, n)` distance matrix: memory is `O(n)` and results are unchanged.
- KFCCL stores the kernel matrix only up to 5000 samples (at most 200 MB);
  above that it computes kernel rows on demand, using `O(K n)` memory at the
  cost of recomputing them every epoch. Both paths give the same result to
  rounding.
- Run-time type checking (typeguard) was measured: on fits of seven
  feature-matrix estimators with 200 and 5000 samples its cost was within
  timing noise, so it stays enabled. `python -O` disables it.
- The README poster was recompressed losslessly (4.6 MB to 0.74 MB,
  pixel-identical).

### Documentation

- One page per estimator, generated from the registry and the docstring at
  build time; the hand-written pages, which contained wrong import paths,
  class names and parameters, were removed. Every estimator's docstring now
  states its objective or update rules, parameters, attributes, reference
  and deviations.
- New user guide: the protocol, what each kind of membership means, choosing
  an algorithm, verification status, reproducibility, relation to
  scikit-learn.
- README rebuilt; its code is executed in testing.
- All 42 examples rewritten: seeded, standalone, and run in the test suite.

### Fixed — algorithms (earlier in this release)

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
  out-of-sample rule set `_supports_out_of_sample` (see the audit section
  above for the list).
- `predict` previously had three incompatible signatures across the library
  (`predict(self)`, `predict(self, X)`, `predict(self, X=None)`), so
  `model.predict(X)` was not uniformly callable. All estimators now use
  `predict(self, X=None)`.
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
