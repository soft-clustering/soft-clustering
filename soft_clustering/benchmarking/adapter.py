"""
BenchmarkAdapter — presents any model to the benchmarking backends through one
``fit(X)`` / ``predict(X)`` / ``membership_`` contract.

Every estimator in this library inherits from
:class:`~soft_clustering._base.BaseSoftClusterer`, which already reconciles the
differing fit signatures and publishes canonical fitted attributes
(``memberships_``, ``labels_``, ``centers_``, ``n_clusters``). The adapter
therefore does **not** re-derive any of that: it delegates to the protocol and
reads those attributes. SCPP estimators can equally be handed to
:class:`~soft_clustering.benchmarking.ClusteringBenchmark` unwrapped.

Two things the adapter is still for:

  1. Labelling a model in a results table — four estimators are exported under
     an alias (``FCM`` is the class ``FuzzyCMeans``), so ``name=`` lets a report
     use the public API name.

  2. Wrapping a model that is *not* an SCPP estimator, such as a scikit-learn
     clusterer. For those the adapter falls back to inspecting the fit
     signature and searching the attribute names SCPP knows about.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from .._base import BaseSoftClusterer, _as_2d_array

# Single source of truth. These tuples live on the estimator protocol; keeping
# private copies here is how the two drifted apart before — this module knew
# five membership attributes while _base.py knew ten.
_MEMBERSHIP_ATTRS: tuple[str, ...] = BaseSoftClusterer._membership_attrs
_CENTER_ATTRS: tuple[str, ...] = BaseSoftClusterer._centers_attrs


class BenchmarkAdapter:
    """
    Wraps a model for use with the benchmarking framework.

    Parameters
    ----------
    model : Any
        An instantiated (but unfitted) model. Any SCPP estimator works; so does
        a foreign estimator exposing ``fit`` or ``fit_predict``.
    n_clusters : int, optional
        Number of clusters. SCPP estimators accept ``n_clusters`` in their
        constructor and supply it to fit themselves, so this is only needed for
        a foreign model whose ``fit_predict(X, K)`` demands it positionally.
    name : str, optional
        Label for this model in benchmark reports. Defaults to the wrapped
        model's class name. Four estimators are exported under an alias
        (``FCM`` is ``FuzzyCMeans``, ``GK`` is ``GustafsonKessel``, ``GMM`` is
        ``GaussianMixtureEM``, ``PCM`` is ``PossibilisticCMeans``); pass
        ``name`` to report the public API name instead of the class name.

    Attributes
    ----------
    name : str
        The label above, so that benchmark reports identify the estimator
        rather than this wrapper. See ``base.model_name``.

    Attributes set after fit()
    --------------------------
    membership_ : ndarray of shape (n_samples, n_clusters), or None
    centers_    : ndarray of shape (n_clusters, n_features), or None
    labels_     : ndarray of shape (n_samples,), or None
    """

    def __init__(
        self,
        model: Any,
        n_clusters: int | None = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.n_clusters = n_clusters
        self.name = name if name is not None else type(model).__name__
        self.membership_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Interface required by the benchmarking backends
    # ------------------------------------------------------------------

    def fit(self, X) -> BenchmarkAdapter:
        """Fit the wrapped model and populate the standardised attributes."""
        if isinstance(self.model, BaseSoftClusterer):
            self._fit_scpp(X)
        else:
            self._fit_foreign(X)
        return self

    def predict(self, X=None) -> np.ndarray:
        if self.labels_ is None:
            raise RuntimeError("BenchmarkAdapter: call fit() before predict().")
        return self.labels_

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Avoid infinite recursion for the adapter's own attrs
        if name in (
            "model",
            "n_clusters",
            "name",
            "membership_",
            "centers_",
            "labels_",
        ):
            raise AttributeError(name)
        return getattr(self.model, name)

    # ------------------------------------------------------------------
    # SCPP estimators: the protocol has already done the work
    # ------------------------------------------------------------------

    def _fit_scpp(self, X) -> None:
        if self.n_clusters is not None and self.model.n_clusters is None:
            # A count passed here rather than to the constructor; the fit
            # wrapper in _base picks it up from the instance.
            self.model.n_clusters = self.n_clusters
        self.model.fit(X)
        self.membership_ = self.model.memberships_
        self.centers_ = self.model.centers_
        self.labels_ = self.model.labels_

    # ------------------------------------------------------------------
    # Foreign models: detect the interface, then normalise
    # ------------------------------------------------------------------

    def _fit_foreign(self, X) -> None:
        result = self._dispatch_fit(X)
        self._extract_membership(result, X)
        self._extract_centers()

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
                        f"{type(self.model).__name__}.fit_predict() requires "
                        "K as a positional argument. "
                        "Pass n_clusters= to BenchmarkAdapter."
                    )
                return self.model.fit_predict(X, self.n_clusters)
            return self.model.fit_predict(X)

        if hasattr(self.model, "fit"):
            self.model.fit(X)
            return None

        raise TypeError(
            f"{type(self.model).__name__} exposes neither fit_predict() nor fit()."
        )

    def _extract_membership(self, result, X) -> None:
        """
        Pull the membership matrix from the fit result or model attributes
        and normalise it to shape (n_samples, n_clusters).
        """
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        U: np.ndarray | None = None

        # --- 1. Try the return value of fit_predict ---
        if result is not None:
            candidates = result if isinstance(result, tuple) else (result,)
            for item in candidates:
                array = _as_2d_array(item)
                if array is not None:
                    U = array
                    break
            if U is None:
                # No 2-D result; a 1-D result is a hard label vector.
                for item in candidates:
                    if isinstance(item, np.ndarray) and item.ndim == 1:
                        self.labels_ = item
                        break

        # --- 2. Try the attribute names the protocol knows about ---
        for attr in _MEMBERSHIP_ATTRS:
            value = _as_2d_array(getattr(self.model, attr, None))
            if value is None:
                continue
            # Transpose if stored as (n_clusters, n_samples)
            if value.shape[0] != n_samples and value.shape[1] == n_samples:
                value = value.T
            if value.shape[0] == n_samples:
                U = value
                break

        self.membership_ = U
        if U is not None and self.labels_ is None:
            self.labels_ = np.argmax(U, axis=1)

    def _extract_centers(self) -> None:
        for attr in _CENTER_ATTRS:
            value = getattr(self.model, attr, None)
            if isinstance(value, np.ndarray):
                self.centers_ = value
                return
        self.centers_ = None
