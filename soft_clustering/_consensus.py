# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Shared input handling for the soft consensus (ensemble) estimators."""

from __future__ import annotations

import numpy as np

from ._validation import check_array


def check_ensemble(soft_memberships, estimator: str) -> list[np.ndarray]:
    """Validate a list of soft membership matrices over the same samples.

    Returns a list of float64 copies; the caller's matrices are never
    modified. Rows are not required to sum to one (possibilistic inputs are
    accepted), but every entry must be finite and non-negative.
    """
    if isinstance(soft_memberships, np.ndarray) and soft_memberships.ndim == 2:
        soft_memberships = [soft_memberships]
    try:
        matrices = list(soft_memberships)
    except TypeError as exc:
        raise TypeError(
            f"{estimator}: expected a list of (n_samples, n_clusters_q) membership "
            f"matrices; got {type(soft_memberships).__name__}."
        ) from exc
    if not matrices:
        raise ValueError(f"{estimator}: the ensemble is empty.")
    out = []
    for q, M in enumerate(matrices):
        M = check_array(M, estimator=estimator, name=f"soft_memberships[{q}]")
        if np.any(M < 0):
            raise ValueError(
                f"{estimator}: soft_memberships[{q}] has negative entries."
            )
        out.append(M)
    n = {M.shape[0] for M in out}
    if len(n) != 1:
        raise ValueError(
            f"{estimator}: all membership matrices must cover the same samples; "
            f"got row counts {sorted(n)}."
        )
    return out


def one_hot(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    U = np.zeros((labels.shape[0], n_clusters))
    U[np.arange(labels.shape[0]), labels] = 1.0
    return U
