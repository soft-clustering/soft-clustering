# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""
Benchmarking utilities for SCPP.

A unified framework for evaluating soft clustering algorithms with respect to
runtime, memory consumption, scalability, and clustering quality, together with
the dataset loaders and cluster-validity metrics the evaluation needs.

The suite ships inside the installed package, so it is available directly after
``pip install soft-clustering``. Running a benchmark additionally needs pandas,
psutil and scikit-learn::

    pip install "soft-clustering[bench]"

Example
-------
>>> from soft_clustering import FCM
>>> from soft_clustering.benchmarking import (
...     BenchmarkAdapter,
...     ClusteringBenchmark,
...     RuntimeBenchmark,
...     get_dataset,
... )
>>> X, y = get_dataset("iris")
>>> models = [BenchmarkAdapter(FCM(random_state=0), n_clusters=3, name="FCM")]
>>> ClusteringBenchmark(models, [RuntimeBenchmark()]).run(X, y)  # doctest: +SKIP
"""

from __future__ import annotations

# Public name -> defining submodule. Imports are deferred (PEP 562), matching
# the parent package, so that importing this module costs nothing and does not
# pull in pandas, psutil or scikit-learn until a feature that needs them is
# actually touched.
_EXPORTS = {
    # Core framework
    "BenchmarkAdapter": ".adapter",
    "BaseBenchmark": ".base",
    "ClusteringBenchmark": ".runner",
    "BenchmarkReport": ".report",
    # Benchmark backends
    "ClusteringQualityBenchmark": ".benchmark.clustering_quality",
    "MemoryBenchmark": ".benchmark.memory_usage",
    "RuntimeBenchmark": ".benchmark.runtime",
    "ScalabilityBenchmark": ".benchmark.scalability",
    # Datasets
    "available_datasets": ".datasets",
    "available_groups": ".datasets",
    "benchmark_suite": ".datasets",
    "dataset_info": ".datasets",
    "datasets_in_group": ".datasets",
    "get_dataset": ".datasets",
    # Metrics
    "clustering_metrics": ".metrics",
    "soft_clustering_metrics": ".metrics",
}


def __getattr__(name):
    """Import a benchmarking export on first access (PEP 562)."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    obj = getattr(import_module(module_path, __name__), name)
    globals()[name] = obj  # cache, so __getattr__ runs once per export
    return obj


def __dir__():
    return sorted(list(globals()) + list(_EXPORTS))


__all__ = sorted(_EXPORTS)
