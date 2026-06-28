from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
try:
    import scipy.sparse as sp
    _HAVE_SCIPY = True
except ImportError:
    sp = None  # type: ignore[assignment]
    _HAVE_SCIPY = False

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

from ..base import BaseBenchmark

# All attribute names where soft_clustering models store the membership matrix.
# Checked in priority order; the first 2-D ndarray found wins.
_MEMBERSHIP_ATTRS = (
    "membership_",         # BenchmarkAdapter (standardised)
    "memberships_",        # FCM, GK
    "typicalities_",       # PCM
    "responsibilities_",   # GMM
    "membership_matrix",   # PFCM (may be c×n — handled below)
    "U",                   # CAFCM, KFCM (KFCM: may be K×n — handled below)
)


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
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

        model.fit(X)

        # ----------------------------------------------------------------
        # Obtain hard labels
        # ----------------------------------------------------------------
        labels: Optional[np.ndarray] = None

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
                f"{model.__class__.__name__}. "
                "Wrap the model with BenchmarkAdapter."
            )

        # ----------------------------------------------------------------
        # Hard clustering metrics
        # ----------------------------------------------------------------
        results: Dict[str, float] = {}
        n_clusters = int(len(np.unique(labels)))

        if n_clusters > 1 and _HAVE_SKLEARN:
            results["silhouette"] = float(
                silhouette_score(X, labels)
            )
            results["calinski_harabasz"] = float(
                calinski_harabasz_score(X, labels)
            )
            results["davies_bouldin"] = float(
                davies_bouldin_score(X, labels)
            )

        if y is not None and _HAVE_SKLEARN:
            results["ari"] = float(
                adjusted_rand_score(y, labels)
            )
            results["nmi"] = float(
                normalized_mutual_info_score(y, labels)
            )

        # ----------------------------------------------------------------
        # Soft clustering metrics
        # ----------------------------------------------------------------
        U = _find_membership(model, n_samples)
        if U is not None:
            results["partition_coefficient"] = float(
                np.mean(np.sum(U ** 2, axis=1))
            )
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
) -> Optional[np.ndarray]:
    """
    Search model attributes for a soft membership matrix of shape
    (n_samples, n_clusters).  Returns None if none is found.
    """
    for attr in _MEMBERSHIP_ATTRS:
        val = getattr(model, attr, None)
        if val is None:
            continue
        if _HAVE_SCIPY and sp.issparse(val):
            val = val.toarray()
        if not (isinstance(val, np.ndarray) and val.ndim == 2):
            continue
        # Transpose if stored as (n_clusters, n_samples)
        if val.shape[0] != n_samples and val.shape[1] == n_samples:
            val = val.T
        if val.shape[0] == n_samples:
            return val
    return None
