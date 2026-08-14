# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""
Third-party packages the benchmarking suite needs but the estimators do not.

``pandas``, ``psutil`` and ``scikit-learn`` are required only to *run* a
benchmark — never to fit a model — so they stay out of the package's runtime
dependencies and ``pip install soft-clustering`` remains lightweight. Importing
them here defensively means ``import soft_clustering.benchmarking`` succeeds on
a bare install; the directed error is raised at the point of use instead, when
the user actually reaches for the feature that needs the dependency.

Modules import the (possibly ``None``) handle for use in type annotations,
which ``from __future__ import annotations`` keeps unevaluated, and call the
matching ``require_*`` helper before touching the library at runtime.
"""

from __future__ import annotations

from typing import Any

BENCH_EXTRA_HINT = (
    'install the benchmarking extras with: pip install "soft-clustering[bench]"'
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only without the extra
    pd = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only without the extra
    psutil = None  # type: ignore[assignment]


def _require(module: Any, package: str, feature: str) -> Any:
    if module is None:
        raise ImportError(
            f"{feature} requires the optional dependency '{package}'; "
            f"{BENCH_EXTRA_HINT}"
        )
    return module


def require_pandas(feature: str) -> Any:
    """Return the ``pandas`` module, or raise a directed ImportError."""
    return _require(pd, "pandas", feature)


def require_psutil(feature: str) -> Any:
    """Return the ``psutil`` module, or raise a directed ImportError."""
    return _require(psutil, "psutil", feature)
