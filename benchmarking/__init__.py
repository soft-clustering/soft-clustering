"""
Benchmarking utilities for SCPP.

This module provides a unified framework for evaluating
soft clustering algorithms with respect to runtime,
memory consumption, scalability, and clustering quality.
"""

from .adapter import BenchmarkAdapter
from .benchmark.clustering_quality import ClusteringQualityBenchmark
from .benchmark.memory_usage import MemoryBenchmark
from .benchmark.runtime import RuntimeBenchmark
from .benchmark.scalability import ScalabilityBenchmark
from .runner import ClusteringBenchmark

__all__ = [
    "ClusteringBenchmark",
    "RuntimeBenchmark",
    "MemoryBenchmark",
    "ScalabilityBenchmark",
    "ClusteringQualityBenchmark",
    "BenchmarkAdapter",
]
