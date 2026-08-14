# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""The common estimator protocol shared by every SCPP soft clustering model.

Soft clustering algorithms differ irreducibly in what they consume: prototype
methods take a feature matrix ``X``, graph methods take an adjacency or
affinity matrix, ensemble methods take a list of membership matrices, and
document methods take raw text. SCPP does not hide that difference — the
first argument of ``fit`` is whatever the algorithm actually needs.

What SCPP does standardize is everything downstream of the fit call. After
``fit`` returns, every estimator exposes:

``memberships_``
    ``ndarray`` of shape ``(n_samples, n_clusters)``, non-negative. Rows sum
    to one whenever the underlying formulation imposes the partition
    constraint; see :attr:`BaseSoftClusterer._partition_constrained`.

``labels_``
    ``ndarray`` of shape ``(n_samples,)``, the arg-max over ``memberships_``.

``centers_``
    ``ndarray`` of shape ``(n_clusters, n_features)`` for prototype-based
    methods, ``None`` otherwise.

``n_clusters``
    The number of clusters the model was fitted with.

Subclasses opt in by inheriting from :class:`BaseSoftClusterer` and declaring
where their solution lives::

    @typechecked
    class FuzzyCMeans(BaseSoftClusterer):
        _k_param = None                      # K arrives at fit time
        _membership_attrs = ("memberships_",)
        _centers_attrs = ("centers_",)

No algorithm body needs to change. ``__init_subclass__`` wraps the estimator's
own entry point (``fit`` or ``fit_predict``, whichever it defines) and
normalises the results afterwards, so legacy call signatures keep working
while the canonical attributes are always populated.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import numpy as np

try:  # SciPy is a hard dependency, but keep the import defensive for clarity.
    import scipy.sparse as sp

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    sp = None
    _HAVE_SCIPY = False


# Constructor keywords that historically stood for "number of clusters".
# Accepted as aliases of ``n_clusters`` so that existing user code keeps
# working, and so that one name is enough to drive every estimator.
K_ALIASES: tuple[str, ...] = (
    "n_clusters",
    "c",
    "k",
    "K",
    "n_topics",
    "n_components",
    "n_communities",
    "c_clusters",
)


def _n_samples_of(data: Any) -> int | None:
    """Best-effort sample count for the heterogeneous inputs SCPP accepts."""
    if data is None:
        return None
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    # Ensemble methods receive a list of per-partition membership matrices;
    # the sample count is their shared first dimension, not the list length.
    if isinstance(data, (list, tuple)) and data:
        first = data[0]
        first_shape = getattr(first, "shape", None)
        if first_shape is not None and len(first_shape) >= 1:
            return int(first_shape[0])
    try:
        return len(data)
    except TypeError:
        return None


def _as_2d_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if _HAVE_SCIPY and sp.issparse(value):
        value = value.toarray()
    if not isinstance(value, np.ndarray):
        return None
    if value.ndim != 2:
        return None
    return value


class BaseSoftClusterer:
    """Base class establishing the SCPP soft clustering protocol.

    Class attributes
    ----------------
    _k_param : str or None
        Name of the constructor parameter this estimator uses for the cluster
        count. ``None`` means the count is supplied at fit time instead.
    _membership_attrs : tuple of str
        Attribute names to search, in priority order, for the membership
        matrix produced by the algorithm.
    _centers_attrs : tuple of str
        Attribute names to search for cluster prototypes.
    _partition_constrained : bool
        ``True`` when the formulation requires each row of ``memberships_`` to
        sum to one. Possibilistic and typicality-based methods set this to
        ``False``; metrics that assume normalisation consult it rather than
        silently producing meaningless numbers.
    """

    _k_param: str | None = "n_clusters"

    # Defaults cover the naming conventions actually used across the library,
    # so most estimators need no per-class declaration; unusual ones override.
    _membership_attrs: tuple[str, ...] = (
        "memberships_",  # FCM, GK
        "typicalities_",  # PCM
        "responsibilities_",  # GMM
        "membership_matrix",  # PFCM (stored clusters-first)
        "membership",  # SoftDBSCANGM
        "resp",  # MBMM
        "P_z_given_d",  # PLSI (stored topics-first)
        "U",  # CAFCM, CAFHFCM, ENTROPYFCM, KFCM, ...
        "W",  # BIGCLAM, BayesianNMF
        "F",  # NOCD
    )
    _centers_attrs: tuple[str, ...] = (
        "centers_",  # FCM, PCM, GK
        "means_",  # GMM
        "centroids",  # CAFCM and the annealing family
        "cluster_centroids",  # PFCM
        "V",  # KFCM
    )
    _partition_constrained: bool = True

    #: ``True`` for methods that discover the number of clusters themselves
    #: (density-, threshold-, or merge-based). For these, ``n_clusters`` is a
    #: fitted attribute rather than a hyperparameter, and any value passed to
    #: the constructor is ignored.
    _determines_k: bool = False

    # Populated by the fit wrapper installed in __init_subclass__.
    memberships_: np.ndarray | None = None
    labels_: np.ndarray | None = None
    centers_: np.ndarray | None = None
    n_clusters: int | None = None

    # ------------------------------------------------------------------
    # Subclass instrumentation
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._install_init_alias()
        cls._install_fit_wrapper()

    @classmethod
    def _install_init_alias(cls) -> None:
        """Let every estimator accept ``n_clusters`` regardless of its own spelling."""
        init = cls.__dict__.get("__init__")
        if init is None or getattr(init, "_scpp_wrapped", False):
            return

        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):  # pragma: no cover
            return

        # The parameter this estimator actually declares for the cluster count.
        own = next((a for a in K_ALIASES if a in params), None)
        cls._k_param = own

        @functools.wraps(init)
        def wrapper(self, *args, **kw):
            # Translate whichever spelling of the cluster count the caller used
            # into the one this estimator declares. ``n_clusters`` is the
            # documented name; the others are accepted for backward
            # compatibility with code written against the original signatures.
            if own is not None:
                for alias in K_ALIASES:
                    if alias != own and alias in kw:
                        kw.setdefault(own, kw.pop(alias))
                        break
            # Accept a cluster count even when the estimator takes it at fit
            # time or discovers it; it is then used as the default for fit().
            deferred = None
            if own is None:
                for alias in K_ALIASES:
                    if alias in kw:
                        deferred = kw.pop(alias)
                        break
            init(self, *args, **kw)
            if own is not None:
                self.n_clusters = getattr(self, own, kw.get(own))
                if self.n_clusters is None:
                    bound = inspect.signature(init).bind_partial(self, *args, **kw)
                    self.n_clusters = bound.arguments.get(own)
            elif deferred is not None:
                self.n_clusters = deferred

        wrapper._scpp_wrapped = True  # type: ignore[attr-defined]
        cls.__init__ = wrapper  # type: ignore[assignment]

    @classmethod
    def _install_fit_wrapper(cls) -> None:
        """Normalise the estimator's own entry point without rewriting it."""
        for name in ("fit", "fit_predict"):
            func = cls.__dict__.get(name)
            if func is None or getattr(func, "_scpp_wrapped", False):
                continue

            try:
                params = list(inspect.signature(func).parameters.values())
            except (TypeError, ValueError):  # pragma: no cover
                params = []
            # Does the entry point demand K positionally, e.g. fit_predict(X, K)?
            k_positional = next(
                (
                    p.name
                    for p in params[1:]
                    if p.name in K_ALIASES and p.default is inspect.Parameter.empty
                ),
                None,
            )

            @functools.wraps(func)
            def wrapper(self, *args, __func=func, __k=k_positional, __name=name, **kw):
                # Supply K from the constructor when the legacy signature wants
                # it positionally and the caller did not provide it.
                if __k is not None and __k not in kw:
                    provided = len(args) - 1  # args[0] is the data
                    if provided < 1 and self.n_clusters is not None:
                        kw[__k] = self.n_clusters
                result = __func(self, *args, **kw)
                self._normalise_fitted_state(result, args[0] if args else kw.get("X"))
                # fit() returns self; fit_predict() returns the memberships.
                if __name == "fit":
                    return self
                return self.memberships_ if result is None else result

            wrapper._scpp_wrapped = True  # type: ignore[attr-defined]
            setattr(cls, name, wrapper)

    # ------------------------------------------------------------------
    # Canonical interface
    # ------------------------------------------------------------------

    def fit(self, X, *args, **kwargs) -> BaseSoftClusterer:
        """Fit the estimator and return ``self``.

        Estimators that define their own ``fit`` override this; those that
        only define ``fit_predict`` inherit it.
        """
        self.fit_predict(X, *args, **kwargs)
        return self

    def fit_predict(self, X, *args, **kwargs) -> np.ndarray:
        """Fit the estimator and return ``memberships_``."""
        self.fit(X, *args, **kwargs)
        return self.memberships_

    def predict(self, X=None) -> np.ndarray:
        """Return hard cluster assignments.

        Most soft clustering algorithms in SCPP are transductive: they produce
        a partition of the data they were fitted on and define no out-of-sample
        rule. For those, ``predict`` ignores ``X`` and returns the stored
        ``labels_``; estimators with a genuine out-of-sample rule override it.
        """
        self._check_fitted()
        return self.labels_

    def predict_proba(self, X=None) -> np.ndarray:
        """Return the membership matrix, with the same caveat as :meth:`predict`."""
        self._check_fitted()
        return self.memberships_

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.memberships_ is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit() first."
            )

    def _normalise_fitted_state(self, result: Any, data: Any) -> None:
        """Populate ``memberships_``, ``labels_``, ``centers_`` after a fit."""
        n_samples = _n_samples_of(data)
        U = self._extract_membership(result, n_samples)

        if U is not None:
            self.memberships_ = U
            if U.shape[1] == 0:
                # Degenerate partition: a non-parametric method found no
                # cluster (e.g. every feature was filtered out). Report every
                # sample as unassigned rather than raising from argmax.
                self.labels_ = np.full(U.shape[0], -1, dtype=int)
            else:
                self.labels_ = np.argmax(U, axis=1)
            # After fitting, ``n_clusters`` reports the partition that was
            # actually produced. For non-parametric methods (:attr:`_determines_k`)
            # this is the discovered count, which may differ from any requested
            # value; reporting the request would be misleading.
            self.n_clusters = int(U.shape[1])

        centers = self._extract_centers()
        if centers is not None:
            self.centers_ = centers

    def _extract_membership(self, result: Any, n_samples: int | None):
        """Find the membership matrix in the return value or on the instance."""
        candidates = []

        if isinstance(result, tuple):
            candidates.extend(result)
        elif result is not None:
            candidates.append(result)

        for attr in self._membership_attrs:
            candidates.append(getattr(self, attr, None))

        for candidate in candidates:
            U = _as_2d_array(candidate)
            if U is None:
                continue
            # Several algorithms store the matrix as (n_clusters, n_samples).
            if n_samples is not None:
                if U.shape[0] != n_samples and U.shape[1] == n_samples:
                    U = U.T
                if U.shape[0] != n_samples:
                    continue
            return np.asarray(U, dtype=np.float64)

        # Last resort: consensus and ensemble methods (SCSPA, SHBGF, SMCLA)
        # return a crisp consensus partition. Represent it as the equivalent
        # one-hot membership matrix so that downstream evaluation code needs no
        # special case; the partition is genuinely crisp, not softened here.
        for candidate in candidates:
            labels = candidate
            if not isinstance(labels, np.ndarray) or labels.ndim != 1:
                continue
            if not np.issubdtype(labels.dtype, np.integer):
                continue
            if n_samples is not None and labels.shape[0] != n_samples:
                continue
            classes = np.unique(labels)
            if classes.size == 0:
                continue
            onehot = np.zeros((labels.shape[0], int(classes.max()) + 1))
            onehot[np.arange(labels.shape[0]), labels] = 1.0
            return onehot
        return None

    def _extract_centers(self):
        for attr in self._centers_attrs:
            value = getattr(self, attr, None)
            if isinstance(value, np.ndarray) and value.ndim == 2:
                return value
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        k = self.n_clusters
        return f"{type(self).__name__}(n_clusters={k})"
