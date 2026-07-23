from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseBenchmark


class RuntimeBenchmark(BaseBenchmark):
    """
    Benchmark training and inference runtime.
    """

    name = "runtime"

    def __init__(
        self,
        n_repeats: int = 3,
    ):
        self.n_repeats = n_repeats

    def evaluate(
        self,
        model: Any,
        X,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        fit_times = []
        predict_times = []

        for _ in range(self.n_repeats):

            start = time.perf_counter()
            model.fit(X)
            fit_times.append(time.perf_counter() - start)

            if hasattr(model, "predict"):

                start = time.perf_counter()

                try:
                    model.predict(X)
                except Exception:
                    pass

                predict_times.append(time.perf_counter() - start)

        return {
            "fit_time_sec": float(np.mean(fit_times)),
            "fit_time_std": float(np.std(fit_times)),
            "predict_time_sec": (
                float(np.mean(predict_times)) if predict_times else np.nan
            ),
        }
