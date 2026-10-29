# Benchmarks

Scripts that regenerate every benchmark result committed in `results/`.
Nothing here is part of the installed package; the measurement code they use
is `soft_clustering.benchmarking`.

```bash
pip install -e ".[bench,baselines]"

# Cross-family comparison of the 21 feature-matrix estimators
python benchmarks/run_main_benchmark.py                      # real + synthetic, seeds 0 1 2
python benchmarks/run_main_benchmark.py --groups real synthetic openml --seeds 0 1 2 3 4
python benchmarks/run_main_benchmark.py --list-modalities    # which estimators take what

# Agreement with, and runtime against, independent implementations
python benchmarks/run_external_baselines.py --sizes 200 1000 5000 --repeats 3
```

## What each script does

**`run_main_benchmark.py`** fits every estimator that takes a plain feature
matrix on every dataset of the chosen groups, once per seed, each fit in its
own subprocess with a timeout (default 60 s). Failures, timeouts and
degenerate partitions are recorded with their status, never dropped. It
scores each run with the external indices (ARI, NMI), the internal indices
(silhouette, Calinski–Harabasz, Davies–Bouldin) and the fuzzy indices, and
writes

| File | Content |
| --- | --- |
| `results/main_benchmark.csv` | one row per (estimator, dataset, seed) |
| `results/main_benchmark_summary.csv` | per-estimator means over completed runs |
| `results/main_benchmark.tex` | the summary as a LaTeX table |
| `results/main_benchmark_env.json` | provenance: platform, library versions, seeds, the SCPP commit (marked `+uncommitted-changes` if the package source differed from it), and datasets that could not be loaded |

Design choices that affect the numbers:

- Every stochastic estimator is seeded (`random_state` = the `seed` column).
- The suite is not standardised. Kernel bandwidths, and ENTROPYFCM's
  temperature, are set by one scale heuristic for every dataset
  (`1 / (d * mean feature variance)`, as scikit-learn's `gamma="scale"`).
  FCC's `jnd`, a perceptual threshold in CIELAB units by default, is set to
  a tenth of the RMS spread `sqrt(d * mean feature variance)`; with the
  default of 20 it merged every point into one cluster on most of the suite.
  ECM uses its full focal-set lattice up to 12 clusters and singletons plus
  the whole frame above that (`olivetti_faces` has 40).
  These are choices of the benchmark, not of the methods.
- `olivetti_faces` has 4096 features and 400 samples. Estimators that keep a
  full covariance per cluster (GK, Gath–Geva, GMM, SoftDBSCANGM) cannot
  allocate it and are recorded as errors; that is a property of those
  methods in high dimensions, not a benchmark failure.
- A run whose fit succeeded but whose scoring raised is recorded as
  `metric_error` with the reason.
- Completed runs are successful, non-degenerate results whose scoring succeeded
  (`ok` or `slow`); means include only their finite measurements. The console and
  LaTeX table show aggregate **Failures**, including timeouts. In the summary CSV,
  `errors` retains its narrower meaning (`error` plus `construct_error`), while
  `timeouts`, `crashes`, `empty_partitions`, `metric_errors`, and `other_failures`
  give the rest of the disjoint breakdown. Their sum is `failures`, and
  `runs = completed + degenerate + failures`. Unknown statuses count as
  `other_failures`.
- Datasets larger than `--max-samples` (2000) are subsampled with a fixed
  seed.
- Fuzziness indices (partition coefficient, entropy) measure crispness, not
  quality, and are never used to rank estimators.
- `olivetti_faces` (group `real`) and the `openml` group are downloaded on
  first use; an unreachable dataset is skipped and listed in the
  provenance file. If scikit-learn's download of `olivetti_faces` fails (its
  figshare mirror has answered HTTP 403), the OpenML copy (data id 41083) is
  used; it has the same shape, pixel range and classes.
- Graph, document, ensemble, image, multi-view and semi-supervised
  estimators take other inputs and are not part of this comparison.

**`run_external_baselines.py`** compares FCM with scikit-fuzzy, GMM and LDA
with scikit-learn, and ECM with evclust (the authors' package, started from
the same prototypes), reporting the maximum membership difference, the
prototype gap, the adjusted Rand index between the partitions and the
runtime of both. Writes `results/external_baselines.{csv,tex}` and
`_env.json`.

## Reproducibility

The correctness columns (memberships, ARI, NMI, statuses) are deterministic
given the seed, the platform's floating-point behaviour and library versions,
which the provenance file records. Runtimes and memory are properties of the
machine and of what else was running; compare them only within one run. Do
not run other heavy jobs while a benchmark is running.

## History

The results committed before the 0.2.0 correctness audit were produced by
implementations that the audit found to be wrong for several estimators
(see `CHANGELOG.md`). They were regenerated with the corrected code; the
earlier files remain in the git history.
