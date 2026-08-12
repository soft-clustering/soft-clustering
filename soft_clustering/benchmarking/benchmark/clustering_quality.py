from __future__ import annotations

from typing import Any

import numpy as np

from ..._base import BaseSoftClusterer, _as_2d_array
from ..base import BaseBenchmark, model_name
from ..metrics import clustering_metrics, is_partition_constrained
from ..metrics import soft_clustering_metrics as _soft_metrics

# Checked in priority order; the first 2-D array of the right shape wins.
# "membership_" is what BenchmarkAdapter publishes; the rest come from the
# estimator protocol, so this list cannot drift out of step with _base.py.
_MEMBERSHIP_ATTRS: tuple[str, ...] = (
    "membership_",
) + BaseSoftClusterer._membership_attrs

_CENTER_ATTRS: tuple[str, ...] = ("centers_",) + BaseSoftClusterer._centers_attrs


class ClusteringQualityBenchmark(BaseBenchmark):
    """Compute clustering quality metrics for one fitted model.

    Supports models wrapped with :class:`BenchmarkAdapter` as well as any
    model that exposes ``fit(X)`` directly.

    The metric definitions live in :mod:`soft_clustering.benchmarking.metrics`
    and are called from here rather than reimplemented, so a benchmark row and
    a direct call to :func:`soft_clustering_metrics` cannot disagree.

    Hard metrics (``nan`` when the partition is degenerate):
        ``silhouette``, ``calinski_harabasz``, ``davies_bouldin``

    Supervised metrics (``nan`` when ``y`` is not provided):
        ``ari``, ``nmi``

    Soft metrics (``nan`` when no membership matrix or no prototypes are
    exposed):
        ``partition_coefficient``, ``modified_partition_coefficient``,
        ``partition_entropy``, ``fuzzy_hypervolume``, ``xie_beni``,
        ``fuzzy_compactness``, ``fuzzy_separation``

    The three fuzziness indices are reported as ``nan`` for estimators that do
    not impose the partition constraint --- possibilistic and affiliation
    based methods --- because they are not defined on unnormalised
    memberships. Whether the constraint holds is taken from the estimator's
    ``_partition_constrained`` flag when it has one, and detected from the
    membership matrix otherwise.
    """

    name = "quality"

    def __init__(self, m: float = 2.0):
        """
        Parameters
        ----------
        m : float, default=2.0
            Fuzzifier used to weight memberships in the prototype-based
            indices.
        """
        self.m = m

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

        if hasattr(model, "labels_") and model.labels_ is not None:
            labels = np.asarray(model.labels_)

        if labels is None:
            U = _find_membership(model, n_samples)
            if U is not None:
                labels = np.argmax(U, axis=1)

        if labels is None:
            raise ValueError(
                f"Cannot obtain cluster labels from {model_name(model)}. "
                "Wrap the model with BenchmarkAdapter."
            )

        results = clustering_metrics(X, labels, y_true=y)

        # ----------------------------------------------------------------
        # Soft clustering metrics
        # ----------------------------------------------------------------
        U = _find_membership(model, n_samples)
        if U is None:
            nan = float("nan")
            results.update(
                {
                    "partition_coefficient": nan,
                    "modified_partition_coefficient": nan,
                    "partition_entropy": nan,
                    "fuzzy_hypervolume": nan,
                    "xie_beni": nan,
                    "fuzzy_compactness": nan,
                    "fuzzy_separation": nan,
                }
            )
        else:
            results.update(
                _soft_metrics(
                    np.asarray(X, dtype=np.float64),
                    U,
                    centers=_find_centers(model, U.shape[1], np.shape(X)[1]),
                    m=self.m,
                    partition_constrained=_declared_constraint(model, U),
                )
            )

        return results


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _find_membership(model: Any, n_samples: int) -> np.ndarray | None:
    """Search model attributes for an ``(n_samples, n_clusters)`` membership matrix."""
    for attr in _MEMBERSHIP_ATTRS:
        val = _as_2d_array(getattr(model, attr, None))
        if val is None:
            continue
        # Transpose if stored as (n_clusters, n_samples).
        if val.shape[0] != n_samples and val.shape[1] == n_samples:
            val = val.T
        if val.shape[0] == n_samples:
            return np.asarray(val, dtype=np.float64)
    return None


def _find_centers(model: Any, n_clusters: int, n_features: int) -> np.ndarray | None:
    """Search model attributes for ``(n_clusters, n_features)`` prototypes."""
    for attr in _CENTER_ATTRS:
        val = _as_2d_array(getattr(model, attr, None))
        if val is None:
            continue
        if val.shape == (n_clusters, n_features):
            return np.asarray(val, dtype=np.float64)
        if val.shape == (n_features, n_clusters):
            return np.asarray(val.T, dtype=np.float64)
    return None


def _declared_constraint(model: Any, U: np.ndarray) -> bool:
    """Prefer the estimator's declaration, fall back to inspecting ``U``.

    A ``BenchmarkAdapter`` forwards the flag from the estimator it wraps.
    """
    declared = getattr(model, "_partition_constrained", None)
    if declared is None:
        return is_partition_constrained(U)
    # A declaration of True that the matrix contradicts is a bug in the
    # estimator; trust the data so the reported numbers stay meaningful.
    return bool(declared) and is_partition_constrained(U)
