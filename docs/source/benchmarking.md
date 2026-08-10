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

Estimators in this library expose three different fit interfaces. The adapter
detects which applies by inspecting the signature, calls the right method, and
normalises the result to a `membership_` matrix of shape
`(n_samples, n_clusters)`:

| Interface | Example estimators | `n_clusters` |
| --- | --- | --- |
| `fit_predict(X, K)` | `FCM`, `PCM`, `GK`, `GMM` | **required** |
| `fit_predict(X)` | `CAFCM`, `AFCM`, `FCC`, `RPFKM`, `RoughKMeans` | from `__init__` |
| `fit(X)` | `KFCM`, `KFCCL`, `ECM`, `SCM`, `MBMM`, `PFCM` | from `__init__` |

It also transposes membership matrices stored as `(n_clusters, n_samples)`,
densifies sparse ones, and locates cluster centres across the various attribute
names the estimators use (`centers_`, `means_`, `centroids`, …).

```{note}
Four estimators are exported under an alias — `FCM` is the class
`FuzzyCMeans`, `GK` is `GustafsonKessel`, `GMM` is `GaussianMixtureEM`, and
`PCM` is `PossibilisticCMeans`. Reports label a model by its class name unless
you pass `name=`, so pass `name="FCM"` if you want result tables to use the
public API name.
```

## Benchmark backends

| Backend | Reports |
| --- | --- |
| `RuntimeBenchmark` | `fit_time_sec`, `fit_time_std`, `predict_time_sec` |
| `MemoryBenchmark` | `memory_before_mb`, `memory_after_mb`, `memory_delta_mb`, `peak_memory_mb` |
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
