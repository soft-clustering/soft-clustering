"""
Benchmarking utilities for SCPP.

This module provides a unified framework for evaluating
soft clustering algorithms with respect to runtime,
memory consumption, scalability, and clustering quality.
"""

from .runner import ClusteringBenchmark
from .benchmark.runtime import RuntimeBenchmark
from .benchmark.memory_usage import MemoryBenchmark
from .benchmark.scalability import ScalabilityBenchmark
from .benchmark.clustering_quality import ClusteringQualityBenchmark
from .adapter import BenchmarkAdapter

__all__ = [
    "ClusteringBenchmark",
    "RuntimeBenchmark",
    "MemoryBenchmark",
    "ScalabilityBenchmark",
    "ClusteringQualityBenchmark",
    "BenchmarkAdapter",
]
