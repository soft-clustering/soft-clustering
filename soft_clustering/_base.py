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
    ``ndarray`` of shape ``(n_samples,)``, the arg-max over ``memberships_``,
    or ``-1`` for a sample whose memberships are all zero (possible only for
    estimators without the partition constraint).

``centers_``
    ``ndarray`` of shape ``(n_clusters, n_features)`` for prototype-based
    methods, ``None`` otherwise.

``n_clusters_``
    The number of clusters actually produced. For methods that discover the
    count it may differ from any requested ``n_clusters``, which is never
    modified by ``fit``.

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
import numbers
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

try:  # SciPy is a hard dependency, but keep the import defensive for clarity.
    import scipy.sparse as sp

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    sp = None
    _HAVE_SCIPY = False


try:  # scikit-learn is a hard dependency; the guard keeps the base importable.
    from sklearn.exceptions import NotFittedError as _SklearnNotFittedError
except ImportError:  # pragma: no cover
    _SklearnNotFittedError = ValueError


class NotFittedError(RuntimeError, _SklearnNotFittedError):
    """Raised when a fitted attribute or method is used before ``fit``.

    Subclasses both :class:`RuntimeError`, which SCPP raised before, and
    scikit-learn's ``NotFittedError``, so code catching either keeps working.
    """


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
    "n_blocks",
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
        # One-dimensional elements are feature rows, not ensemble matrices.
        if first_shape is not None and len(first_shape) == 2:
            return int(first_shape[0])
    try:
        return len(data)
    except TypeError:
        return None


def ratio_memberships(distances: np.ndarray, exponent: float) -> np.ndarray:
    """The fuzzy c-means ratio rule, evaluated without overflow.

    Returns the ``(n_samples, n_clusters)`` matrix

    .. math::

        u_{ik} \\;=\\; \\frac{d_{ik}^{-p}}{\\sum_j d_{ij}^{-p}}
               \\;=\\; \\Big[\\sum_j (d_{ik}/d_{ij})^{p}\\Big]^{-1},

    where ``exponent`` is :math:`p`, already matched to whether ``distances``
    holds distances (:math:`p = 2/(m-1)`) or squared distances
    (:math:`p = 1/(m-1)`).

    Why this exists as a shared helper. Written directly as
    ``d ** -p / (d ** -p).sum(...)``, the rule overflows whenever a sample
    lies on a prototype (``d`` underflows, ``d ** -p`` is ``inf``) or the
    fuzzifier approaches one (``p`` grows without bound), and the
    normalisation then yields an all-``NaN`` row. Nine modules in this library
    each carried their own copy of the unstable form.

    The fix is to factor out the per-sample minimum distance before
    exponentiating. That factor cancels exactly between numerator and
    denominator, so the value is unchanged, while every ratio is at least one
    and every power therefore lies in ``(0, 1]``. A sample coincident with a
    prototype underflows to a clean one-hot row instead of producing ``NaN``.
    Evaluating in log space keeps that true for any ``p``.
    """
    distances = np.asarray(distances, dtype=np.float64)
    log_d = np.log(np.maximum(distances, np.finfo(np.float64).tiny))
    scores = -exponent * (log_d - log_d.min(axis=1, keepdims=True))
    np.exp(scores, out=scores)
    total = scores.sum(axis=1, keepdims=True)
    return scores / np.where(total > 0, total, 1.0)


def _labels_from(U: np.ndarray) -> np.ndarray:
    """Hard labels: ``argmax`` of each row, or ``-1`` for an all-zero row.

    Methods without the partition constraint (typicality-, affiliation- and
    threshold-based ones) can leave a sample with no membership at all.
    ``argmax`` would silently put such a sample in cluster 0; ``-1`` marks it
    as unassigned, the convention scikit-learn uses for noise.
    """
    labels = np.argmax(U, axis=1)
    if U.shape[1]:
        labels[np.max(U, axis=1) <= 0] = -1
    return labels


def _bind_params(init: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Constructor arguments as a name -> value mapping, defaults included."""
    try:
        bound = inspect.signature(init).bind(None, *args, **kwargs)
    except TypeError:  # pragma: no cover - the call itself would have failed
        return {}
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    # Drop ``self`` and any *args / **kwargs catch-alls, which are not
    # reconstructible parameters.
    for name, parameter in inspect.signature(init).parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            arguments.pop(name, None)
    first = next(iter(inspect.signature(init).parameters), None)
    if first is not None:
        arguments.pop(first, None)
    return arguments


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

    #: ``True`` only for estimators that define a genuine out-of-sample rule,
    #: i.e. that can assign a sample they were not fitted on. Most soft
    #: clustering algorithms are transductive and cannot; for those,
    #: ``predict(X)`` raises rather than silently returning the training
    #: labels, which would be a partition of the wrong data with the wrong
    #: length.
    _supports_out_of_sample: bool = False

    #: Sample count recorded by estimators whose input is not one array.
    _n_samples_hint: int | None = None

    # Populated by the fit wrapper installed in __init_subclass__. They are
    # None until then; estimator code reads them only after fitting (and
    # ``_check_fitted`` enforces that for public methods), so they are typed
    # as fitted for the type checker. ``n_clusters`` is None for estimators
    # that have no such parameter.
    if TYPE_CHECKING:
        memberships_: np.ndarray
        labels_: np.ndarray
        centers_: np.ndarray
        n_clusters: int
        n_clusters_: int
    else:
        memberships_ = None
        labels_ = None
        centers_ = None
        n_clusters = None
        n_clusters_ = None

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
            # Record the constructor arguments verbatim so that get_params
            # round-trips exactly. Reading them back off the instance would
            # not: several estimators store a parameter under a different
            # attribute name (PFCM keeps ``typicality_fuzzifier`` as ``eta``).
            self._scpp_params = _bind_params(init, args, kw)
            if own is not None:
                self.n_clusters = getattr(self, own, kw.get(own))
                if self.n_clusters is None:
                    bound = inspect.signature(init).bind_partial(self, *args, **kw)
                    self.n_clusters = bound.arguments.get(own)
            elif deferred is not None:
                self.n_clusters = deferred
                # The estimator takes K at fit time, so it is not one of the
                # constructor's own parameters; record it anyway or clone()
                # would silently drop it.
                self._scpp_params["n_clusters"] = deferred

        wrapper._scpp_wrapped = True  # type: ignore[attr-defined]
        cls.__init__ = wrapper  # type: ignore[method-assign]

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
            # Unsupervised single-input entry points, fit(X) or the legacy
            # fit_predict(X, K=None), follow scikit-learn's convention of
            # accepting and ignoring ``y``: Pipelines and model-selection tools
            # call fit(X, y). Estimators whose second argument means something
            # (two views, partial labels, an image shape, ...) are left alone.
            data_params = [
                p
                for p in params[1:]
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            # An optional ``y`` is scikit-learn's ignored target; a required
            # one (SFCMEP's partial labels) is real input and is passed on.
            explicit_y = (
                len(data_params) >= 2
                and data_params[1].name == "y"
                and data_params[1].default is not inspect.Parameter.empty
            )
            rest = data_params[2:] if explicit_y else data_params[1:]
            legacy_k = (
                rest[0].name
                if len(rest) == 1
                and rest[0].name in K_ALIASES
                and rest[0].default is not inspect.Parameter.empty
                else None
            )
            unsupervised = len(rest) == 0 or legacy_k is not None
            # Does the entry point demand K positionally, e.g. fit_predict(X, K)?
            k_positional = next(
                (
                    p.name
                    for p in params[1:]
                    if p.name in K_ALIASES and p.default is inspect.Parameter.empty
                ),
                None,
            )

            # Loop variables are bound through defaults; mypy cannot type the
            # resulting signature against functools.wraps.
            @functools.wraps(func)  # type: ignore[arg-type]
            def wrapper(
                self,
                *args,
                __func=func,
                __k=k_positional,
                __name=name,
                __unsup=unsupervised,
                __legacy_k=legacy_k,
                **kw,
            ):
                # Every fit starts from the constructor's parameters, not from
                # whatever a previous fit left behind: ``n_clusters`` is
                # overwritten with the discovered count after a fit, and stale
                # fitted attributes must not leak into a failed refit.
                self._reset_fitted_state()
                if __unsup:
                    args, kw = self._absorb_y(args, kw, __legacy_k)
                # Supply K from the constructor when the legacy signature wants
                # it positionally and the caller did not provide it.
                if __k is not None and __k not in kw:
                    provided = len(args) - 1  # args[0] is the data
                    if provided < 1 and self.n_clusters is not None:
                        kw[__k] = self.n_clusters
                result = __func(self, *args, **kw)
                self._normalise_fitted_state(result, args[0] if args else kw.get("X"))
                # fit() returns self; fit_predict() returns the memberships,
                # whatever the algorithm's own body returned. Earlier releases
                # passed that through, so fit_predict returned a tuple, a dict,
                # a sparse matrix or hard labels depending on the estimator.
                if __name == "fit":
                    return self
                return self.memberships_

            if unsupervised and not explicit_y and legacy_k is None:
                # Advertise scikit-learn's fit(X, y=None) signature.
                sig = inspect.signature(func)
                new_params = list(sig.parameters.values())
                new_params.insert(
                    2,
                    inspect.Parameter(
                        "y", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    ),
                )
                wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
                    parameters=new_params
                )
            wrapper._scpp_wrapped = True  # type: ignore[attr-defined]
            setattr(cls, name, wrapper)  # noqa: B010

    # ------------------------------------------------------------------
    # Canonical interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None, *args, **kwargs) -> BaseSoftClusterer:
        """Fit the estimator and return ``self``.

        Estimators that define their own ``fit`` override this; those that
        only define ``fit_predict`` inherit it. For unsupervised estimators
        ``y`` is ignored, as in scikit-learn; for estimators whose second input
        is meaningful (partial labels, a second view, ...) it is passed on.
        """
        if y is None and not args:
            self.fit_predict(X, **kwargs)
        else:
            self.fit_predict(X, y, *args, **kwargs)
        return self

    def fit_predict(self, X, y=None, *args, **kwargs) -> np.ndarray:
        """Fit the estimator and return ``memberships_``.

        ``y`` is ignored by unsupervised estimators, as in scikit-learn, and
        passed on to estimators whose second input is meaningful.
        """
        if y is None and not args:
            self.fit(X, **kwargs)
        else:
            self.fit(X, y, *args, **kwargs)
        return self._fitted_memberships()

    def predict(self, X=None) -> np.ndarray:
        """Return hard cluster assignments.

        Called with no argument, returns ``labels_``, the partition of the
        data the model was fitted on.

        Called with new data ``X``, returns ``argmax`` of
        :meth:`predict_proba(X) <predict_proba>` for estimators that define an
        out-of-sample rule (``_supports_out_of_sample = True``), such as the
        prototype-based fuzzy and possibilistic methods, whose membership
        update is a closed-form function of the fitted prototypes. Most other
        soft clustering algorithms in SCPP are transductive: they partition
        the data they were fitted on and define no rule for a new sample.
        For those, passing ``X`` raises :class:`NotImplementedError`, because
        returning the training labels would silently hand back a partition of
        different data, of the wrong length.
        """
        self._check_fitted()
        if X is None:
            assert self.labels_ is not None  # set together with memberships_
            return self.labels_
        self._reject_out_of_sample(X, "predict")
        return _labels_from(self._predict_memberships(X))

    def predict_proba(self, X=None) -> np.ndarray:
        """Return memberships; see :meth:`predict` for the meaning of ``X``.

        With no argument this is ``memberships_``. With new data it applies
        the estimator's membership rule to ``X`` against the fitted model
        parameters, without refitting them.
        """
        if X is None:
            return self._fitted_memberships()
        self._check_fitted()
        self._reject_out_of_sample(X, "predict_proba")
        return self._predict_memberships(X)

    def _predict_memberships(self, X) -> np.ndarray:
        """Membership rule for unseen data; overridden by inductive estimators."""
        raise NotImplementedError  # pragma: no cover - guarded by the flag

    def _reject_out_of_sample(self, X, method: str) -> None:
        if X is None or self._supports_out_of_sample:
            return
        raise NotImplementedError(
            f"{type(self).__name__} is transductive: it partitions the data it "
            f"was fitted on and defines no out-of-sample rule, so "
            f"{method}(X) is not available. Call {method}() with no argument "
            f"for the fitted partition, or refit on the combined data."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _absorb_y(self, args: tuple, kw: dict, legacy_k: str | None):
        """Drop scikit-learn's ``y`` from an unsupervised fit call.

        A second positional argument used to be the cluster count for FCM,
        PCM, GK and GMM (``fit_predict(X, K)``). An integer there is still
        honoured, with a deprecation warning; anything else is ``y`` and is
        ignored, as for every unsupervised scikit-learn estimator.
        """
        kw.pop("y", None)
        if len(args) > 2:
            raise TypeError(
                f"{type(self).__name__}.fit takes the data and an optional, "
                f"ignored y; got {len(args)} positional arguments."
            )
        if len(args) == 2:
            extra = args[1]
            args = args[:1]
            if (
                legacy_k is not None
                and isinstance(extra, numbers.Integral)
                and not isinstance(extra, bool)
            ):
                warnings.warn(
                    "Passing the number of clusters to fit is deprecated; set "
                    "n_clusters in the constructor instead.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                kw[legacy_k] = int(extra)
        return args, kw

    def __sklearn_is_fitted__(self) -> bool:
        return self.memberships_ is not None

    def _reset_fitted_state(self) -> None:
        """Clear the protocol attributes before a fit."""
        for name in ("memberships_", "labels_", "centers_", "n_clusters_"):
            setattr(self, name, None)  # back to "not fitted"
        self._n_samples_hint = None
        self.__dict__.pop("n_features_in_", None)

    def _fitted_memberships(self) -> np.ndarray:
        """``memberships_``, or a clear error if the estimator is not fitted."""
        if self.memberships_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call "
                "fit() first."
            )
        return self.memberships_

    def _check_fitted(self) -> None:
        if self.memberships_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call "
                "fit() first."
            )

    def _normalise_fitted_state(self, result: Any, data: Any) -> None:
        """Populate ``memberships_``, ``labels_``, ``centers_`` after a fit."""
        # Sample count is normally read off the fit input, but for a few
        # estimators the input is not sample-major -- an image clusterer
        # receives an (H, W) array and partitions H*W pixels. Those set
        # ``_n_samples_hint`` during fit and it wins.
        n_samples = getattr(self, "_n_samples_hint", None)
        if n_samples is None:
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
                self.labels_ = _labels_from(U)
            # The partition actually produced. For methods that discover the
            # count (:attr:`_determines_k`) it can differ from any requested
            # value. The ``n_clusters`` parameter itself is left unchanged, as
            # scikit-learn requires of parameters; earlier releases overwrote it.
            self.n_clusters_ = int(U.shape[1])

        centers = self._extract_centers()
        if centers is not None:
            self.centers_ = centers

        shape = getattr(data, "shape", None)
        if shape is not None and len(shape) == 2:
            self.n_features_in_ = int(shape[1])

    def _extract_membership(self, result: Any, n_samples: int | None):
        """Find the membership matrix in the return value or on the instance."""
        candidates: list[Any] = []

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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor arguments, scikit-learn style.

        Implementing this (with :meth:`set_params`) is what makes
        ``sklearn.base.clone`` and scikit-learn's parameter introspection work
        on SCPP estimators. It does **not** make them drop-in scikit-learn
        clusterers: ``fit`` signatures differ by input modality and no
        estimator defines ``score``, so tools that require a uniform
        ``fit(X, y)`` or a default scorer --- ``GridSearchCV`` without an
        explicit ``scoring`` argument, for one --- still do not apply.

        Parameters
        ----------
        deep : bool
            Accepted for signature compatibility. SCPP estimators do not
            contain nested estimators, so it has no effect.
        """
        snapshot = getattr(self, "_scpp_params", None)
        if snapshot is None:  # pragma: no cover - estimator with no own __init__
            return {}
        # As in scikit-learn, a parameter's value is the attribute of the same
        # name, so ``model.n_clusters = 4`` is a parameter change. The
        # constructor snapshot only covers a parameter that is not stored as
        # an attribute.
        return {
            name: getattr(self, name) if hasattr(self, name) else value
            for name, value in snapshot.items()
        }

    def set_params(self, **params: Any) -> BaseSoftClusterer:
        """Set constructor arguments in place and return ``self``.

        The estimator is reconstructed from the merged parameters, so any
        derived state the constructor computes stays consistent. Fitted
        attributes are discarded, as they are for a scikit-learn estimator.
        """
        current = self.get_params()
        unknown = sorted(set(params) - set(current))
        if unknown:
            raise ValueError(
                f"invalid parameter(s) {unknown} for estimator "
                f"{type(self).__name__}; valid parameters are "
                f"{sorted(current)}"
            )
        current.update(params)
        type(self).__init__(self, **current)
        return self

    def __sklearn_tags__(self):
        """Estimator tags for scikit-learn >= 1.6.

        Declares SCPP estimators as unsupervised clusterers, so that
        scikit-learn utilities that query tags (``is_clusterer``, pipelines,
        ``check_estimator``) can inspect them. Only defined in terms of the
        public tag API; older scikit-learn versions do not call it.
        """
        from sklearn.base import BaseEstimator

        tags = BaseEstimator().__sklearn_tags__()
        tags.estimator_type = "clusterer"
        tags.target_tags.required = False
        tags.non_deterministic = False
        # Sparse feature matrices are accepted and densified.
        tags.input_tags.sparse = True
        return tags

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = self.get_params()
        if not params:
            return f"{type(self).__name__}()"
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"{type(self).__name__}({rendered})"
