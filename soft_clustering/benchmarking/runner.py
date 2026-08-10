from __future__ import annotations

from typing import Any, Optional

from ._optional import pd, require_pandas
from .base import BaseBenchmark, model_name


class ClusteringBenchmark:
    """
    Main benchmarking interface for SCPP.
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
        X,
        y: Optional = None,
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
