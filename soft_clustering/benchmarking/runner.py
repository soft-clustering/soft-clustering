from __future__ import annotations

from typing import Any

import numpy as np

from ._optional import pd, require_pandas
from .base import BaseBenchmark, model_name


class ClusteringBenchmark:
    """
    Main benchmarking interface for SCPP.

    Runs every benchmark against every model and collects the results into one
    row per model.

    Parameters
    ----------
    models : list
        Fitted-or-unfitted estimators. SCPP estimators can be passed directly;
        :class:`~soft_clustering.benchmarking.BenchmarkAdapter` is only needed
        to relabel a model or to wrap a non-SCPP one.
    benchmarks : list of BaseBenchmark
        The measurements to take.
    """

    def __init__(
        self,
        models: list[Any],
        benchmarks: list[BaseBenchmark],
    ):
        self.models = models
        self.benchmarks = benchmarks

    def run(
        self,
        X: Any,
        y: np.ndarray | None = None,
    ) -> pd.DataFrame:

        require_pandas("ClusteringBenchmark.run()")

        records = []

        for model in self.models:

            row = {"model": model_name(model)}

            for benchmark in self.benchmarks:

                result = benchmark.evaluate(model=model, X=X, y=y)

                row.update(result)

            records.append(row)

        return pd.DataFrame(records)
