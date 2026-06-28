"""
BenchmarkAdapter — normalises the heterogeneous soft_clustering model
interfaces into the single fit(X) / predict(X) / membership_ contract
expected by the benchmarking backends.

Soft clustering models in this library come in three variants:

  1. fit_predict(X, K)   — K is a required positional argument at call time
                           (FCM, PCM, GK, GMM)

  2. fit_predict(X)      — K was given to __init__; no K needed at call time
                           (CAFCM, AFCM, AFCMSimple, CAFHFCM, ENTROPYFCM,
                            FCC, RPFKM, RoughKMeans, RDFKC)

  3. fit(X)              — sklearn-style; K given to __init__
                           (KFCM, KFCCL, ECM, SCM, SoftDBSCANGM, MBMM, PFCM)

The adapter auto-detects which case applies via signature inspection,
calls the right method, then extracts a standardised membership_ matrix
of shape (n_samples, n_clusters).
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np

try:
    import scipy.sparse as sp
    _HAVE_SCIPY = True
except ImportError:
    sp = None  # type: ignore[assignment]
    _HAVE_SCIPY = False


# Attribute names where models store the soft membership matrix, in lookup
# priority order.  The adapter checks each in turn.
_MEMBERSHIP_ATTRS = (
    "memberships_",        # FCM, GK
    "typicalities_",       # PCM
    "responsibilities_",   # GMM
    "membership_matrix",   # PFCM  (stored as c×n — transposed)
    "U",                   # CAFCM, KFCM  (KFCM: stored as K×n — transposed)
)

# Attribute names where models store cluster prototypes / centres.
_CENTER_ATTRS = (
    "centers_",            # FCM, PCM, GK
    "means_",              # GMM
    "centroids",           # CAFCM
    "cluster_centroids",   # PFCM
    "V",                   # KFCM  (n_clusters × n_features — normal)
)


class BenchmarkAdapter:
    """
    Wraps any soft_clustering model for use with the benchmarking framework.

    Parameters
    ----------
    model : Any
        An instantiated (but unfitted) soft_clustering model.
    n_clusters : int, optional
        Number of clusters.  **Required** for models whose fit_predict(X, K)
        signature demands K as a positional argument (FCM, PCM, GK, GMM).
        Ignored (but harmless) for models that already carry K in __init__.

    Attributes set after fit()
    --------------------------
    membership_ : ndarray of shape (n_samples, n_clusters), or None
    centers_    : ndarray of shape (n_clusters, n_features), or None
    labels_     : ndarray of shape (n_samples,)
    """

    def __init__(self, model: Any, n_clusters: Optional[int] = None) -> None:
        self.model = model
        self.n_clusters = n_clusters
        self.membership_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # sklearn-compatible interface required by the benchmarking backends
    # ------------------------------------------------------------------

    def fit(self, X) -> "BenchmarkAdapter":
        """Fit the wrapped model and populate standardised attributes."""
        result = self._dispatch_fit(X)
        self._extract_membership(result, X)
        self._extract_centers()
        return self

    def predict(self, X) -> np.ndarray:
        if self.labels_ is None:
            raise RuntimeError("BenchmarkAdapter: call fit() before predict().")
        return self.labels_

    # ------------------------------------------------------------------
    # Class-name transparency (so ClusteringBenchmark reports the real name)
    # ------------------------------------------------------------------

    @property
    def __class__(self):  # type: ignore[override]
        return self.model.__class__

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Avoid infinite recursion for the adapter's own attrs
        if name in ("model", "n_clusters", "membership_", "centers_", "labels_"):
            raise AttributeError(name)
        return getattr(self.model, name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dispatch_fit(self, X):
        """
        Detect the model's interface and call the appropriate method.

        Decision order:
          1. fit_predict(X, K)  — K is a required positional argument
          2. fit_predict(X)     — K was supplied to __init__
          3. fit(X)             — sklearn-style
        """
        if hasattr(self.model, "fit_predict"):
            sig = inspect.signature(self.model.fit_predict)
            params = list(sig.parameters.values())
            # params[0] is 'X'; params[1] (if present) may be K
            needs_K = (
                len(params) >= 2
                and params[1].name in ("K", "k", "n_clusters")
                and params[1].default is inspect.Parameter.empty
            )
            if needs_K:
                if self.n_clusters is None:
                    raise ValueError(
                        f"{self.model.__class__.__name__}.fit_predict() requires "
                        "K as a positional argument. "
                        "Pass n_clusters= to BenchmarkAdapter."
                    )
                return self.model.fit_predict(X, self.n_clusters)
            else:
                return self.model.fit_predict(X)

        if hasattr(self.model, "fit"):
            self.model.fit(X)
            return None

        raise TypeError(
            f"{self.model.__class__.__name__} exposes neither "
            "fit_predict() nor fit()."
        )

    def _extract_membership(self, result, X) -> None:
        """
        Pull the membership matrix from the fit result or model attributes
        and normalise it to shape (n_samples, n_clusters).
        """
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        U: Optional[np.ndarray] = None

        # --- 1. Try the return value of fit_predict ---
        if result is not None:
            if _HAVE_SCIPY and sp.issparse(result):
                U = result.toarray()
            elif isinstance(result, np.ndarray) and result.ndim == 2:
                U = result
            elif isinstance(result, tuple):
                # e.g. CAFCM returns (labels, U)
                for item in result:
                    if isinstance(item, np.ndarray) and item.ndim == 2:
                        U = item
                        break
                if U is None:
                    # grab 1-D labels from tuple
                    for item in result:
                        if isinstance(item, np.ndarray) and item.ndim == 1:
                            self.labels_ = item
                            break

        # --- 2. Try well-known model attributes ---
        for attr in _MEMBERSHIP_ATTRS:
            val = getattr(self.model, attr, None)
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
                U = val
                break

        self.membership_ = U
        if U is not None and self.labels_ is None:
            self.labels_ = np.argmax(U, axis=1)

    def _extract_centers(self) -> None:
        for attr in _CENTER_ATTRS:
            val = getattr(self.model, attr, None)
            if val is not None and isinstance(val, np.ndarray):
                self.centers_ = val
                return
        self.centers_ = None
