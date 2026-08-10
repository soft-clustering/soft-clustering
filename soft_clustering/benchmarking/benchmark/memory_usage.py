from __future__ import annotations

import gc
import threading
import time
from typing import Any

import numpy as np

from .._optional import require_psutil
from ..base import BaseBenchmark

_BYTES_PER_MB = 1024**2


class MemoryBenchmark(BaseBenchmark):
    """
    Benchmark memory consumption of a clustering model.

    Resident set size is sampled on a background thread while ``fit`` runs, so
    that transient allocations — an ``n x n`` kernel or distance matrix, say —
    are reflected in ``peak_memory_mb`` rather than being missed because they
    were freed before the fit returned.

    Metrics
    -------
    memory_before_mb :
        Resident set size before fitting.

    memory_after_mb :
        Resident set size after fitting.

    memory_delta_mb :
        Retained memory: ``memory_after_mb - memory_before_mb``. Negative
        values are possible when fitting triggers a collection that frees more
        than the model retains.

    peak_memory_mb :
        Largest resident set size observed, over the samples taken during the
        fit plus the before and after readings.

    n_samples_taken :
        Number of background samples actually collected. Useful for judging
        ``peak_memory_mb``: a fit that returns in less than ``poll_interval``
        yields no samples, and the peak then degenerates to the larger of the
        before and after readings.

    fit_time_sec :
        Runtime of ``model.fit()``.

    Notes
    -----
    Sampling runs in a Python thread, so it observes the process only when the
    fitting code releases the GIL. NumPy and SciPy release it around the
    expensive array operations that dominate these algorithms, which is where
    the allocations of interest occur. Resident set size is also a
    process-wide measure: it attributes to the model any allocation made
    concurrently elsewhere in the process.
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
            Sampling interval in seconds. Must be positive.

        warmup : bool, default=True
            Run garbage collection before benchmarking, so that memory left
            over from a previous model is not attributed to this one.
        """
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive, got {poll_interval!r}")
        self.poll_interval = poll_interval
        self.warmup = warmup

    @staticmethod
    def _memory_mb() -> float:
        psutil = require_psutil("MemoryBenchmark.evaluate()")
        process = psutil.Process()
        return process.memory_info().rss / _BYTES_PER_MB

    def evaluate(
        self,
        model: Any,
        X: Any,
        y: np.ndarray | None = None,
    ) -> dict[str, float]:

        psutil = require_psutil("MemoryBenchmark.evaluate()")
        process = psutil.Process()

        if self.warmup:
            gc.collect()

        memory_before = process.memory_info().rss / _BYTES_PER_MB
        peak = memory_before
        n_samples = 0
        stop = threading.Event()

        def _sample() -> None:
            nonlocal peak, n_samples
            # Event.wait returns True once stop is set, ending the loop without
            # waiting out a further interval.
            while not stop.wait(self.poll_interval):
                try:
                    rss = process.memory_info().rss / _BYTES_PER_MB
                except psutil.Error:  # pragma: no cover - process went away
                    return
                n_samples += 1
                if rss > peak:
                    peak = rss

        sampler = threading.Thread(
            target=_sample,
            name="MemoryBenchmark-sampler",
            daemon=True,
        )
        sampler.start()

        start_time = time.perf_counter()
        try:
            model.fit(X)
        finally:
            end_time = time.perf_counter()
            stop.set()
            sampler.join()

        memory_after = process.memory_info().rss / _BYTES_PER_MB
        peak = max(peak, memory_after)

        # Round the endpoints first and derive the delta from them, so the
        # reported numbers stay consistent with each other: rounding each of
        # the three independently lets the delta disagree with the difference
        # of the reported endpoints by up to a rounding step.
        before_mb = round(memory_before, 3)
        after_mb = round(memory_after, 3)

        return {
            "memory_before_mb": before_mb,
            "memory_after_mb": after_mb,
            "memory_delta_mb": round(after_mb - before_mb, 3),
            "peak_memory_mb": round(peak, 3),
            "n_samples_taken": n_samples,
            "fit_time_sec": round(end_time - start_time, 6),
        }
