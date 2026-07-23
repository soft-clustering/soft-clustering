from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import psutil

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
        process = psutil.Process()
        return process.memory_info().rss / 1024**2

    def evaluate(
        self,
        model: Any,
        X,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

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
