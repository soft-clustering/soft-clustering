# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Input validation and random-state handling shared by the estimators.

Every estimator that consumes a feature matrix routes it through
:func:`check_array`, so malformed input fails the same way everywhere and with
a message that names the estimator and the problem. The checks are
deliberately strict about non-finite values: a NaN that reaches a distance
computation produces a partition that looks valid and is not, which is the
worst failure mode a clustering library can have.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

try:  # scipy is a hard dependency; the guard only keeps the import explicit.
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


RandomStateLike = int | np.random.RandomState | np.random.Generator | None


def check_array(
    X: Any,
    *,
    estimator: str = "estimator",
    name: str = "X",
    ensure_2d: bool = True,
    min_samples: int = 1,
    min_features: int = 1,
    allow_nan: bool = False,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Convert ``X`` to a dense floating-point array and validate it.

    Parameters
    ----------
    X : array-like or scipy sparse matrix
        Input to validate. Sparse input is densified; the estimators that call
        this function operate on dense arrays internally.
    estimator : str
        Name of the calling estimator, used in error messages.
    name : str
        Name of the argument, used in error messages.
    ensure_2d : bool
        Require a two-dimensional ``(n_samples, n_features)`` array.
    min_samples, min_features : int
        Minimum accepted number of rows and columns.
    allow_nan : bool
        Accept NaN entries (infinite values are always rejected). Used by
        estimators that impute missing values themselves.
    dtype : numpy dtype
        Target dtype.

    Returns
    -------
    ndarray
        A C-contiguous array of ``dtype``. The input is never modified.

    Raises
    ------
    ValueError
        If ``X`` has the wrong dimensionality, too few samples or features,
        or contains non-finite values.
        Also raised if ``X`` is complex.
    TypeError
        If ``X`` cannot be converted to a floating-point array.
    """
    if sp is not None and sp.issparse(X):
        X = X.toarray()
    if np.iscomplexobj(X):
        raise ValueError(
            f"{estimator}: Complex data not supported; {name} must be real-valued."
        )
    try:
        array = np.array(X, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{estimator}: {name} must be a numeric array-like; could not convert "
            f"{type(X).__name__} to float ({exc})"
        ) from exc

    if ensure_2d:
        if array.ndim == 1:
            raise ValueError(
                f"{estimator}: {name} must be 2-D with shape (n_samples, "
                f"n_features); got a 1-D array of shape {array.shape}. Reshape "
                "your data either using X.reshape(-1, 1) if it has a single "
                "feature or X.reshape(1, -1) if it contains a single sample."
            )
        if array.ndim != 2:
            raise ValueError(
                f"{estimator}: {name} must be 2-D with shape (n_samples, "
                f"n_features); got shape {array.shape}."
            )
        n_samples, n_features = array.shape
        if n_samples < min_samples:
            raise ValueError(
                f"{estimator}: found array with {n_samples} sample(s) (shape="
                f"{array.shape}) while a minimum of {min_samples} is required."
            )
        if n_features < min_features:
            raise ValueError(
                f"{estimator}: found array with {n_features} feature(s) (shape="
                f"{array.shape}) while a minimum of {min_features} is required."
            )

    if allow_nan:
        if np.isinf(array).any():
            raise ValueError(f"{estimator}: {name} contains infinite values.")
    elif not np.isfinite(array).all():
        n_bad = int(np.size(array) - np.isfinite(array).sum())
        raise ValueError(
            f"{estimator}: {name} contains {n_bad} NaN or infinite value(s). "
            "Clustering is undefined for non-finite input; impute or remove "
            "them first."
        )
    return array


def check_n_features(X: np.ndarray, expected: int, *, estimator: str) -> None:
    """Raise if ``X`` does not have the number of features seen in ``fit``.

    The message follows scikit-learn's wording.
    """
    if X.shape[1] != expected:
        raise ValueError(
            f"X has {X.shape[1]} features, but {estimator} is expecting "
            f"{expected} features as input."
        )


def check_n_clusters(
    n_clusters: Any, n_samples: int | None = None, *, estimator: str = "estimator"
) -> int:
    """Validate a cluster count and return it as ``int``.

    Raises
    ------
    TypeError
        If ``n_clusters`` is not an integer.
    ValueError
        If ``n_clusters < 1`` or ``n_clusters > n_samples``.
    """
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, numbers.Integral):
        raise TypeError(
            f"{estimator}: n_clusters must be an integer; got "
            f"{type(n_clusters).__name__} {n_clusters!r}."
        )
    n_clusters = int(n_clusters)
    if n_clusters < 1:
        raise ValueError(f"{estimator}: n_clusters must be >= 1; got {n_clusters}.")
    if n_samples is not None and n_clusters > n_samples:
        raise ValueError(
            f"{estimator}: n_clusters={n_clusters} exceeds the number of "
            f"samples (n_samples={n_samples})."
        )
    return n_clusters


def check_scalar(
    value: Any,
    name: str,
    *,
    estimator: str = "estimator",
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> float:
    """Validate a real-valued hyperparameter against an interval.

    Raises
    ------
    ValueError
        If ``value`` is not finite or lies outside the interval.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{estimator}: {name} must be a real number; got "
            f"{type(value).__name__} {value!r}."
        )
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{estimator}: {name} must be finite; got {value}.")
    if lower is not None:
        bad = value < lower if lower_inclusive else value <= lower
        if bad:
            op = ">=" if lower_inclusive else ">"
            raise ValueError(f"{estimator}: {name} must be {op} {lower}; got {value}.")
    if upper is not None:
        bad = value > upper if upper_inclusive else value >= upper
        if bad:
            op = "<=" if upper_inclusive else "<"
            raise ValueError(f"{estimator}: {name} must be {op} {upper}; got {value}.")
    return value


def objective_converged(previous: float, current: float, tol: float) -> bool:
    """Scale-free stopping test on an objective value.

    True when ``|previous - current| <= tol * max(|previous|, |current|)``.

    An absolute test ``|previous - current| <= tol`` makes the number of
    iterations depend on the units of the data: the objectives of the
    prototype methods scale with the squared data scale, so expressing the
    same data in kilometres instead of metres made FCM stop after a single
    sweep. The relative test is invariant to that rescaling. An objective that
    is exactly zero on both sides has converged.
    """
    if not (np.isfinite(previous) and np.isfinite(current)):
        return False
    return abs(previous - current) <= tol * max(abs(previous), abs(current))


def data_scale(X: np.ndarray) -> float:
    """Root mean per-feature variance of ``X``: the unit for distance tolerances.

    Stopping rules on prototype movement use ``shift <= tol * data_scale(X)``
    so that, like the objective test above, they do not depend on the units of
    the data. scikit-learn's ``KMeans`` scales its tolerance the same way. A
    constant data set falls back to one, so the test is still well defined.
    """
    scale = float(np.sqrt(np.mean(np.var(X, axis=0)))) if X.size else 0.0
    return scale if scale > 0.0 and np.isfinite(scale) else 1.0


def check_random_state(seed: RandomStateLike) -> np.random.RandomState:
    """Turn ``seed`` into a :class:`numpy.random.RandomState`.

    Follows scikit-learn's convention, which several estimators in this
    library were written against:

    * ``None`` returns NumPy's global ``RandomState`` singleton. Results are
      then not reproducible unless the caller seeds NumPy globally, which is
      exactly the behaviour of the original implementations; it is preserved
      so that the optimisation-equivalence tests can still compare against
      them bit for bit.
    * an ``int`` returns a new, private ``RandomState`` seeded with it. The
      global state is neither read nor modified.
    * a ``RandomState`` is returned unchanged.
    * a ``Generator`` is used to seed a private ``RandomState``.
    """
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return np.random.RandomState(seed.integers(0, 2**32 - 1, dtype=np.uint64))
    if isinstance(seed, numbers.Integral) and not isinstance(seed, bool):
        return np.random.RandomState(int(seed))
    raise ValueError(
        f"random_state must be None, an int, a numpy RandomState or a numpy "
        f"Generator; got {type(seed).__name__} {seed!r}."
    )


def check_generator(seed: RandomStateLike) -> np.random.Generator:
    """Turn ``seed`` into a :class:`numpy.random.Generator`.

    ``None`` gives a freshly entropy-seeded generator (results not
    reproducible, global state untouched); an ``int`` gives a reproducible
    generator; a ``Generator`` is returned unchanged; a ``RandomState`` seeds a
    new generator from its stream.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed.randint(0, 2**31 - 1))
    if seed is None or (
        isinstance(seed, numbers.Integral) and not isinstance(seed, bool)
    ):
        return np.random.default_rng(seed)
    raise ValueError(
        f"random_state must be None, an int, a numpy RandomState or a numpy "
        f"Generator; got {type(seed).__name__} {seed!r}."
    )


def int_seed(random_state: RandomStateLike) -> int | None:
    """An integer seed (for torch or scikit-learn), or ``None`` when unseeded."""
    if random_state is None:
        return None
    if isinstance(random_state, numbers.Integral) and not isinstance(
        random_state, bool
    ):
        return int(random_state)
    return int(check_generator(random_state).integers(0, 2**31 - 1))
