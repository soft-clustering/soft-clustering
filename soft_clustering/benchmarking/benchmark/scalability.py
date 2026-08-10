from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from .._optional import require_psutil
from ..base import BaseBenchmark


class ScalabilityBenchmark(BaseBenchmark):
    """
    Measure runtime and memory as dataset size grows.
    """

    name = "scalability"

    def __init__(
        self,
        sample_sizes: Iterable[int] = (
            100,
            500,
            1000,
            5000,
            10000,
        ),
        random_state: int = 42,
    ):
        self.sample_sizes = list(sample_sizes)
        self.random_state = random_state

    @staticmethod
    def _memory_mb():
        psutil = require_psutil("ScalabilityBenchmark.evaluate()")
        process = psutil.Process()
        return process.memory_info().rss / 1024**2

    def evaluate(
        self,
        model: Any,
        X,
        y: np.ndarray | None = None,
    ) -> dict[str, float]:

        rng = np.random.default_rng(self.random_state)

        results = {}

        for size in self.sample_sizes:

            if size > len(X):
                continue

            idx = rng.choice(
                len(X),
                size=size,
                replace=False,
            )

            X_sub = X[idx]

            mem_before = self._memory_mb()

            start = time.perf_counter()
            model.fit(X_sub)
            runtime = time.perf_counter() - start

            mem_after = self._memory_mb()

            results[f"runtime_{size}"] = runtime

            results[f"memory_{size}"] = mem_after - mem_before

        return results
