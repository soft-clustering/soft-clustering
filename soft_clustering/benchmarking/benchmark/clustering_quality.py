from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

from ..._base import BaseSoftClusterer, _as_2d_array
from ..base import BaseBenchmark, model_name

# Checked in priority order; the first 2-D array of the right shape wins.
# "membership_" is what BenchmarkAdapter publishes; the rest come from the
# estimator protocol, so this list cannot drift out of step with _base.py.
_MEMBERSHIP_ATTRS: tuple[str, ...] = (
    "membership_",
) + BaseSoftClusterer._membership_attrs


class ClusteringQualityBenchmark(BaseBenchmark):
    """
    Compute clustering quality metrics.

    Supports models wrapped with BenchmarkAdapter as well as any model
    that exposes fit(X) directly.

    Hard metrics (always computed when n_clusters > 1):
        silhouette, calinski_harabasz, davies_bouldin

    Supervised metrics (computed when y is provided):
        ari, nmi

    Soft metrics (computed when a membership matrix is found):
        partition_coefficient, partition_entropy
    """

    name = "quality"

    def evaluate(
        self,
        model: Any,
        X,
        y: np.ndarray | None = None,
    ) -> dict[str, float]:

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

        model.fit(X)

        # ----------------------------------------------------------------
        # Obtain hard labels
        # ----------------------------------------------------------------
        labels: np.ndarray | None = None

        if hasattr(model, "predict"):
            try:
                labels = np.asarray(model.predict(X))
            except Exception:
                pass

        if labels is None and hasattr(model, "labels_"):
            labels = np.asarray(model.labels_)

        if labels is None:
            U = _find_membership(model, n_samples)
            if U is not None:
                labels = np.argmax(U, axis=1)

        if labels is None:
            raise ValueError(
                f"Cannot obtain cluster labels from "
                f"{model_name(model)}. "
                "Wrap the model with BenchmarkAdapter."
            )

        # ----------------------------------------------------------------
        # Hard clustering metrics
        # ----------------------------------------------------------------
        results: dict[str, float] = {}
        n_clusters = int(len(np.unique(labels)))

        if n_clusters > 1 and _HAVE_SKLEARN:
            results["silhouette"] = float(silhouette_score(X, labels))
            results["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            results["davies_bouldin"] = float(davies_bouldin_score(X, labels))

        if y is not None and _HAVE_SKLEARN:
            results["ari"] = float(adjusted_rand_score(y, labels))
            results["nmi"] = float(normalized_mutual_info_score(y, labels))

        # ----------------------------------------------------------------
        # Soft clustering metrics
        # ----------------------------------------------------------------
        U = _find_membership(model, n_samples)
        if U is not None:
            results["partition_coefficient"] = float(np.mean(np.sum(U**2, axis=1)))
            results["partition_entropy"] = float(
                np.mean(-np.sum(U * np.log(U + 1e-12), axis=1))
            )

        return results


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _find_membership(
    model: Any,
    n_samples: int,
) -> np.ndarray | None:
    """
    Search model attributes for a soft membership matrix of shape
    (n_samples, n_clusters).  Returns None if none is found.
    """
    for attr in _MEMBERSHIP_ATTRS:
        val = _as_2d_array(getattr(model, attr, None))
        if val is None:
            continue
        # Transpose if stored as (n_clusters, n_samples)
        if val.shape[0] != n_samples and val.shape[1] == n_samples:
            val = val.T
        if val.shape[0] == n_samples:
            return val
    return None
