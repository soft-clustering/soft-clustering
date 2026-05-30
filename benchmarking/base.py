from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaseBenchmark(ABC):
    """
    Base class for all SCPP benchmarks.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X,
        y: Optional = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model and return benchmark results.
        """
        raise NotImplementedError

    def __call__(self, model, X, y=None):
        return self.evaluate(model, X, y)

    @staticmethod
    def validate_result(result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise TypeError(
                "Benchmark results must be returned as a dictionary."
            )

    @staticmethod
    def to_dataframe(results):
        return pd.DataFrame(results)
        