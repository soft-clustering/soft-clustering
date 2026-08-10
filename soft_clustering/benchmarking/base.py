from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ._optional import pd, require_pandas


def model_name(model: Any) -> str:
    """
    Return the name a model should be reported under in a results table.

    ``BenchmarkAdapter`` wraps an estimator and sets ``name`` to the wrapped
    estimator's class name, so that reports identify the real model rather
    than the wrapper. Anything else is reported by its own class name.
    """
    name = getattr(model, "name", None)
    return name if isinstance(name, str) else type(model).__name__


class BaseBenchmark(ABC):
    """
    Base class for all SCPP benchmarks.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X: Any,
        y: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a model and return benchmark results.

        ``X`` is deliberately untyped beyond ``Any``: SCPP estimators accept a
        feature matrix, an adjacency matrix, a list of documents or a list of
        membership matrices, depending on the method.
        """
        raise NotImplementedError

    def __call__(
        self,
        model: Any,
        X: Any,
        y: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return self.evaluate(model, X, y)

    @staticmethod
    def validate_result(result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise TypeError("Benchmark results must be returned as a dictionary.")

    @staticmethod
    def to_dataframe(results) -> pd.DataFrame:
        require_pandas("BaseBenchmark.to_dataframe()")
        return pd.DataFrame(results)
