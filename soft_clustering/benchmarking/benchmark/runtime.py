# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

from ..base import BaseBenchmark


class RuntimeBenchmark(BaseBenchmark):
    """
    Benchmark training and inference runtime.

    Inference time is NaN if any repeat lacks a callable predictor or raises.
    Unsupported inference (NotImplementedError) is silent; other prediction
    exceptions produce a RuntimeWarning. Fit exceptions still propagate.
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
        y: np.ndarray | None = None,
    ) -> dict[str, float]:

        fit_times = []
        predict_times = []

        for _ in range(self.n_repeats):

            start = time.perf_counter()
            model.fit(X)
            fit_times.append(time.perf_counter() - start)

            predict = getattr(model, "predict", None)
            if callable(predict):
                start = time.perf_counter()
                try:
                    predict(X)
                except NotImplementedError:
                    predict_times.append(np.nan)
                except Exception as exc:
                    predict_times.append(np.nan)
                    warnings.warn(
                        f"Prediction timing unavailable: {type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    predict_times.append(time.perf_counter() - start)
            else:
                predict_times.append(np.nan)

        return {
            "fit_time_sec": float(np.mean(fit_times)),
            "fit_time_std": float(np.std(fit_times)),
            "predict_time_sec": (
                float(np.mean(predict_times)) if predict_times else np.nan
            ),
        }
