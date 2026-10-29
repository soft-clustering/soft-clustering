# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Fuzzy c-means and the prototype utilities shared by its relatives."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from typeguard import typechecked

from ._base import BaseSoftClusterer, ratio_memberships
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_n_features,
    check_scalar,
    objective_converged,
)


def _euclidean_dist2(X, centers):
    """Squared Euclidean distances, shape ``(n_samples, n_centers)``.

    Uses compiled direct differences, avoiding cancellation from subtracting
    large squared norms without an (n_samples, n_centers, n_features) temporary.
    """
    return cdist(X, centers, metric="sqeuclidean")


def _normalize_memberships(U, eps=1e-12):
    U = np.maximum(U, eps)
    U_sum = U.sum(axis=1, keepdims=True)
    U /= U_sum + eps
    return U


def _init_centers_kpp(X, K, rng):
    """k-means++ seeding (Arthur and Vassilvitskii, 2007) from data points."""
    n, _ = X.shape
    centers = np.empty((K, X.shape[1]), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    for k in range(1, K):
        dist2 = _euclidean_dist2(X, centers[:k])
        d_min = np.min(dist2, axis=1)
        s = np.sum(d_min)
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(0, n)
        else:
            probs = d_min / s
            idx = rng.choice(n, p=probs)
        centers[k] = X[idx]
    return centers


def _update_U_from_centers(X, centers, m, eps=1e-12):
    """Fuzzy c-means membership update for fixed prototypes."""
    dist2 = _euclidean_dist2(X, centers)
    # ratio_memberships evaluates d^-p / sum_j d^-p without ever forming the
    # unbounded intermediate: written directly, this overflows for m near 1
    # (p = 1/(m-1) grows without bound) and returns an all-NaN row. It also
    # handles a zero distance (a one-hot row), so no additive epsilon is
    # needed; one would make the memberships depend on the units of X.
    U = ratio_memberships(dist2, 1.0 / (m - 1.0))
    return _normalize_memberships(U, eps=eps)


def _update_centers_from_U(X, U, m, eps=1e-12):
    """Weighted means for positive mass; zero vectors for empty clusters.

    ``eps`` is retained for compatibility, but does not bias positive masses.
    """
    Um = U**m
    num = Um.T @ X
    den = np.sum(Um, axis=0, keepdims=True).T
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _objective(X, U, centers, m):
    dist2 = _euclidean_dist2(X, centers)
    return float(np.sum((U**m) * dist2))


def fcm_fit(X, n_clusters, m, rng, *, max_iter=300, tol=1e-5, init="kmeans++"):
    """Run fuzzy c-means; return ``(U, centers, trajectory)``.

    Shared by the estimators that are defined as refinements of a fuzzy
    c-means partition (PCM, PFCM, FeMIFuzzy).
    """
    n = X.shape[0]
    if init == "kmeans++":
        centers = _init_centers_kpp(X, n_clusters, rng)
    elif init == "random":
        U = _normalize_memberships(rng.random((n, n_clusters)))
        centers = _update_centers_from_U(X, U, m)
    else:
        raise ValueError(f"init must be 'kmeans++' or 'random'; got {init!r}.")

    obj_prev = np.inf
    trajectory = []
    U = _update_U_from_centers(X, centers, m)
    for _ in range(max_iter):
        centers = _update_centers_from_U(X, U, m)
        U = _update_U_from_centers(X, centers, m)
        obj = _objective(X, U, centers, m)
        trajectory.append(obj)
        if objective_converged(obj_prev, obj, tol):
            break
        obj_prev = obj
    return U, centers, trajectory


@typechecked
class FuzzyCMeans(BaseSoftClusterer):
    """Fuzzy c-means (FCM).

    Minimises the within-cluster weighted sum of squared distances

    .. math::

        J_m(U, V) = \\sum_{i=1}^{n} \\sum_{k=1}^{K} u_{ik}^m \\lVert x_i - v_k \\rVert^2,
        \\qquad \\sum_k u_{ik} = 1,\\; u_{ik} \\ge 0,

    by alternating the closed-form updates

    .. math::

        u_{ik} = \\Big[\\sum_j \\big(\\lVert x_i - v_k\\rVert^2 /
                 \\lVert x_i - v_j\\rVert^2\\big)^{1/(m-1)}\\Big]^{-1},
        \\qquad
        v_k = \\frac{\\sum_i u_{ik}^m x_i}{\\sum_i u_{ik}^m}.

    Each step cannot increase :math:`J_m`, so ``objective_trajectory_`` is
    non-increasing (up to floating-point rounding).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`K`, ``1 <= K <= n_samples``.
    m : float, default=2.0
        Fuzzifier, ``m > 1``. As ``m -> 1`` the memberships become crisp and
        FCM approaches k-means; large ``m`` drives every membership to
        ``1/K``.
    max_iter : int, default=300
        Maximum number of alternating-optimisation sweeps.
    tol : float, default=1e-5
        Stop when the objective changes by at most ``tol`` relative to its
        magnitude between sweeps, so the test does not depend on the units
        of ``X``.
    init : {"kmeans++", "random"}, default="kmeans++"
        ``"kmeans++"`` seeds the prototypes with k-means++; ``"random"``
        starts from a random membership matrix.
    random_state : int, numpy Generator or None, default=None
        Seed for the initialisation. ``None`` is not reproducible.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy memberships; rows sum to one.
    labels_ : ndarray of shape (n_samples,)
        ``argmax`` of ``memberships_``.
    centers_ : ndarray of shape (n_clusters, n_features)
        Cluster prototypes.
    objective_trajectory_ : ndarray of shape (n_iter,)
        :math:`J_m` after every sweep.
    n_iter_ : int
        Number of sweeps performed.

    References
    ----------
    J. C. Bezdek. *Pattern Recognition with Fuzzy Objective Function
    Algorithms.* Plenum Press, 1981. doi:10.1007/978-1-4757-0450-1

    Examples
    --------
    >>> import numpy as np
    >>> from soft_clustering import FCM
    >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    >>> model = FCM(n_clusters=2, random_state=0).fit(X)
    >>> model.memberships_.shape
    (4, 2)
    >>> bool(model.labels_[0] == model.labels_[1] != model.labels_[2])
    True
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        init: str = "kmeans++",
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def fit_predict(self, X, y=None, K: int | None = None):
        """Fit the model and return the membership matrix.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data. Sparse input is densified.
        y : ignored
            Accepted for scikit-learn compatibility.
        K : int, optional
            Deprecated way of passing the cluster count at fit time; overrides
            ``n_clusters`` when given. Prefer the constructor argument.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
            Fuzzy memberships.
        """
        name = type(self).__name__
        X = check_array(X, estimator=name)
        K = check_n_clusters(
            self.n_clusters if K is None else K, X.shape[0], estimator=name
        )
        m = check_scalar(self.m, "m", estimator=name, lower=1.0, lower_inclusive=False)

        rng = check_generator(self.random_state)
        U, centers, trajectory = fcm_fit(
            X, K, m, rng, max_iter=self.max_iter, tol=self.tol, init=self.init
        )

        self.centers_ = centers
        self.memberships_ = U
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = len(trajectory)
        return self.memberships_

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return _update_U_from_centers(X, self.centers_, self.m)
