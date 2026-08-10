# Benchmarking

`soft_clustering.benchmarking` is a unified framework for evaluating soft
clustering algorithms on equal terms — runtime, memory, scalability, and
clustering quality — across the heterogeneous fit interfaces used by the
estimators in this library.

The suite is part of the installed package, so it is available immediately
after `pip install soft-clustering`. Running a benchmark additionally requires
pandas, psutil and scikit-learn:

```bash
pip install "soft-clustering[bench]"
```

These are deliberately kept out of the core requirements: they are needed to
*run* a benchmark, never to fit a model. `soft_clustering.benchmarking` imports
them defensively, so the module remains importable without them and raises a
directed error only when you reach a feature that needs one.

## Quick start

```python
from soft_clustering import FCM, GK, PCM
from soft_clustering.benchmarking import (
    BenchmarkAdapter,
    BenchmarkReport,
    ClusteringBenchmark,
    ClusteringQualityBenchmark,
    RuntimeBenchmark,
    get_dataset,
)

X, y = get_dataset("iris")

models = [
    BenchmarkAdapter(FCM(random_state=0), n_clusters=3, name="FCM"),
    BenchmarkAdapter(GK(random_state=0), n_clusters=3, name="GK"),
    BenchmarkAdapter(PCM(random_state=0), n_clusters=3, name="PCM"),
]

results = ClusteringBenchmark(
    models=models,
    benchmarks=[RuntimeBenchmark(n_repeats=3), ClusteringQualityBenchmark()],
).run(X, y)

report = BenchmarkReport(results)
print(report.leaderboard("ari"))
report.to_csv("results.csv")
```

## BenchmarkAdapter

Every estimator in this library inherits from `BaseSoftClusterer`, which already
reconciles the differing fit signatures and publishes canonical fitted
attributes (`memberships_`, `labels_`, `centers_`, `n_clusters`). **SCPP
estimators can therefore be benchmarked directly, with no wrapper:**

```python
ClusteringBenchmark(
    models=[FCM(n_clusters=3, random_state=0), KFCM(n_clusters=3)],
    benchmarks=[RuntimeBenchmark()],
).run(X, y)
```

`BenchmarkAdapter` remains useful for two things.

**Labelling a model.** Four estimators are exported under an alias — `FCM` is
the class `FuzzyCMeans`, `GK` is `GustafsonKessel`, `GMM` is
`GaussianMixtureEM`, and `PCM` is `PossibilisticCMeans`. Reports label a model
by its class name, so pass `name=` when you want the public API name in the
results table:

```python
BenchmarkAdapter(FCM(n_clusters=3, random_state=0), name="FCM")
```

**Wrapping a model that is not an SCPP estimator**, such as a scikit-learn
clusterer. For those the adapter inspects the fit signature, calls whichever of
`fit_predict(X, K)`, `fit_predict(X)` or `fit(X)` applies, transposes a
membership matrix stored as `(n_clusters, n_samples)`, densifies a sparse one,
and searches the attribute names the protocol knows about. Pass `n_clusters=`
if the foreign model requires `K` positionally.

```{note}
The attribute names searched are `BaseSoftClusterer._membership_attrs` and
`._centers_attrs` — the benchmarking code reads that registry rather than
keeping its own copy. An estimator that stores its memberships under a new name
should add it there, and every benchmark picks it up.
```

## Benchmark backends

| Backend | Reports |
| --- | --- |
| `RuntimeBenchmark` | `fit_time_sec`, `fit_time_std`, `predict_time_sec` |
| `MemoryBenchmark` | `memory_before_mb`, `memory_after_mb`, `memory_delta_mb`, `peak_memory_mb`, `n_samples_taken` |
| `ScalabilityBenchmark` | `runtime_<n>` and `memory_<n>` at each sample size |
| `ClusteringQualityBenchmark` | silhouette, Calinski-Harabasz, Davies-Bouldin; ARI and NMI when `y` is given; partition coefficient and entropy when a membership matrix is found |

Write your own by subclassing `BaseBenchmark` and implementing
`evaluate(model, X, y)` to return a dictionary of metrics.

## Datasets

`get_dataset(name)` returns `(X, y)` for any registered dataset. Three groups
are available — `real` (bundled with scikit-learn), `synthetic` (generated
locally), and `openml` (downloaded on first use):

```python
from soft_clustering.benchmarking import (
    available_datasets,
    benchmark_suite,
    dataset_info,
    get_dataset,
)

available_datasets()          # every registered name
dataset_info("iris")          # {'name': 'iris', 'n_samples': 150, ...}
benchmark_suite("synthetic")  # {name: (X, y)} for a whole group
```

## Metrics

The validity indices are usable on their own, independently of the benchmark
runner:

```python
from soft_clustering.benchmarking import clustering_metrics, soft_clustering_metrics

soft_clustering_metrics(X, U, centers=centers)
# partition_coefficient, modified_partition_coefficient, partition_entropy,
# fuzzy_hypervolume, xie_beni, fuzzy_compactness, fuzzy_separation

clustering_metrics(X, labels, y_true=y)
# silhouette, calinski_harabasz, davies_bouldin, ari, nmi
```
