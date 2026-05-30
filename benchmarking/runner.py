from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseBenchmark


class ClusteringBenchmark:
    """
    Main benchmarking interface for SCPP.
    """

    def __init__(
        self,
        models: List[Any],
        benchmarks: List[BaseBenchmark],
    ):
        self.models = models
        self.benchmarks = benchmarks

    def run(
        self,
        X,
        y: Optional = None,
    ) -> pd.DataFrame:

        records = []

        for model in self.models:

            model_name = model.__class__.__name__

            row = {
                "model": model_name
            }

            for benchmark in self.benchmarks:

                result = benchmark.evaluate(
                    model=model,
                    X=X,
                    y=y
                )

                row.update(result)

            records.append(row)

        return pd.DataFrame(records)
        