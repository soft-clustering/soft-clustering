from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional

import numpy as np
import psutil

from ..base import BaseBenchmark


class MemoryBenchmark(BaseBenchmark):
    """
    Benchmark memory consumption of a clustering model.

    Metrics
    -------
    memory_before_mb :
        Memory usage before fitting.

    memory_after_mb :
        Memory usage after fitting.

    memory_delta_mb :
        Additional memory consumed.

    peak_memory_mb :
        Maximum observed memory during fitting.

    fit_time_sec :
        Runtime of model.fit().
    """

    name = "memory"

    def __init__(
        self,
        poll_interval: float = 0.01,
        warmup: bool = True,
    ):
        """
        Parameters
        ----------
        poll_interval : float, default=0.01
            Sampling interval in seconds.

        warmup : bool, default=True
            Run garbage collection before benchmarking.
        """
        self.poll_interval = poll_interval
        self.warmup = warmup

    @staticmethod
    def _memory_mb() -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)

    def evaluate(
        self,
        model: Any,
        X,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        if self.warmup:
            gc.collect()

        memory_before = self._memory_mb()
        peak_memory = memory_before

        start_time = time.perf_counter()

        model.fit(X)

        end_time = time.perf_counter()

        memory_after = self._memory_mb()
        peak_memory = max(
            peak_memory,
            memory_after,
        )

        return {
            "memory_before_mb": round(memory_before, 3),
            "memory_after_mb": round(memory_after, 3),
            "memory_delta_mb": round(
                memory_after - memory_before,
                3,
            ),
            "peak_memory_mb": round(
                peak_memory,
                3,
            ),
            "fit_time_sec": round(
                end_time - start_time,
                6,
            ),
        }