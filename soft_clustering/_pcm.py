# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Possibilistic c-means (Krishnapuram and Keller, 1993)."""

from __future__ import annotations

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._fcm import _euclidean_dist2, _init_centers_kpp, fcm_fit
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_n_features,
    check_scalar,
    objective_converged,
)


def _update_T_from_centers(X, centers, etas, m, eps=1e-12):
    """Typicality update ``t_ik = 1 / (1 + (d_ik^2 / eta_k)^{1/(m-1)})``.

    Evaluated as a logistic function of ``log(d^2/eta) / (m - 1)`` so that it
    neither overflows for ``m`` close to one nor returns NaN for a sample
    coincident with a prototype. Positive scales, including subnormals, are
    retained. At zero scale, coincident points have typicality one and other
    points have the dimensionless lower clip ``eps``.
    """
    d2 = _euclidean_dist2(X, centers)
    etas = np.asarray(etas)
    if np.any(etas < 0):
        raise ValueError("PCM: typicality scales must be nonnegative.")
    positive = etas > 0
    distances = d2[:, positive]
    log_distances = np.full_like(distances, -np.inf)
    np.log(distances, out=log_distances, where=distances > 0)
    log_ratio = log_distances - np.log(etas[positive])[None, :]
    z = log_ratio / (m - 1.0)
    # 1 / (1 + exp(z)) without overflow.
    values = np.empty_like(z)
    pos = z >= 0
    ez = np.exp(-z[pos])
    values[pos] = ez / (1.0 + ez)
    values[~pos] = 1.0 / (1.0 + np.exp(z[~pos]))
    T = np.zeros_like(d2)
    T[:, positive] = values
    T[d2 == 0] = 1.0
    return np.clip(T, eps, 1.0)


def _update_centers_from_T(X, T, m, eps=1e-12):
    """Use actual positive weight masses; retain zero-vector empty fallback.

    The legacy ``eps`` argument no longer biases positive-mass means.
    """
    Tm = T**m
    num = Tm.T @ X
    den = Tm.sum(axis=0)[:, None]
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _estimate_etas(X, W, centers, m, alpha=1.0, eps=1e-12):
    """Weighted squared spread (K&K, Eq. 9); zero for exactly empty support.

    ``eps`` is retained for compatibility, without flooring scales or masses.
    """
    d2 = _euclidean_dist2(X, centers)
    Wm = W**m
    num = alpha * (Wm * d2).sum(axis=0)
    den = Wm.sum(axis=0)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _objective_pcm(X, T, centers, etas, m):
    d2 = _euclidean_dist2(X, centers)
    term1 = np.sum((T**m) * d2)
    term2 = np.sum(etas[None, :] * ((1.0 - T) ** m))
    return float(term1 + term2)


@typechecked
class PossibilisticCMeans(BaseSoftClusterer):
    """Possibilistic c-means (PCM).

    Drops the fuzzy partition constraint and penalises low typicality instead:

    .. math::

        J(T, V) = \\sum_{i,k} t_{ik}^m \\lVert x_i - v_k\\rVert^2
                + \\sum_k \\eta_k \\sum_i (1 - t_{ik})^m ,

    giving the closed-form updates

    .. math::

        t_{ik} = \\Big[1 + \\big(\\lVert x_i - v_k\\rVert^2/\\eta_k\\big)^{1/(m-1)}\\Big]^{-1},
        \\qquad v_k = \\frac{\\sum_i t_{ik}^m x_i}{\\sum_i t_{ik}^m}.

    Typicalities are computed independently for each cluster, so rows of
    ``memberships_`` do **not** sum to one: an outlier receives low typicality
    in every cluster.

    Following Krishnapuram and Keller (1993), the run is initialised from a
    fuzzy c-means partition and the scale of each cluster is estimated once
    from it, ``eta_k = alpha * sum_i u_ik^m d_ik^2 / sum_i u_ik^m``, and then
    held fixed. With fixed ``eta`` each update minimises :math:`J` in one
    block of variables, so ``objective_trajectory_`` is non-increasing.

    PCM is known to produce coincident clusters when two prototypes are
    attracted to the same dense region; this is a property of the
    formulation (Barni et al., 1996), not of this implementation, and is one
    motivation for :class:`PFCM`.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    m : float, default=2.0
        Fuzzifier, ``m > 1``.
    alpha : float, default=1.0
        Scale factor ``K`` in the estimate of ``eta_k`` (Krishnapuram and
        Keller use ``K = 1``). Larger values give wider, more tolerant
        clusters.
    max_iter : int, default=300
        Maximum number of sweeps.
    tol : float, default=1e-5
        Stop when the objective changes by at most ``tol`` relative to its
        magnitude.
    init : {"fcm", "kmeans++", "random"}, default="fcm"
        ``"fcm"`` initialises prototypes and ``eta`` from a fuzzy c-means run,
        as the reference prescribes. ``"kmeans++"`` and ``"random"`` seed
        prototypes directly and estimate ``eta`` from the nearest-prototype
        partition; they are provided for ablation.
    random_state : int, numpy Generator or None, default=None
        Seed for the initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Typicalities in ``(0, 1]``; rows need not sum to one.
    typicalities_ : ndarray of shape (n_samples, n_clusters)
        Same array as ``memberships_``.
    centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes.
    etas_ : ndarray of shape (n_clusters,)
        Fixed per-cluster scale parameters.
    objective_trajectory_ : ndarray
        Objective after every sweep.
    n_iter_ : int
        Number of sweeps performed.

    References
    ----------
    R. Krishnapuram and J. M. Keller. A possibilistic approach to clustering.
    *IEEE Transactions on Fuzzy Systems*, 1(2):98--110, 1993.
    doi:10.1109/91.227387
    """

    _partition_constrained = False
    _membership_attrs = ("typicalities_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        alpha: float = 1.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        init: str = "fcm",
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def fit_predict(self, X, y=None, K: int | None = None):
        """Fit the model and return the typicality matrix.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
        y : ignored
            Accepted for scikit-learn compatibility.
        K : int, optional
            Deprecated fit-time cluster count; prefer ``n_clusters``.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
        """
        name = type(self).__name__
        X = check_array(X, estimator=name)
        K = check_n_clusters(
            self.n_clusters if K is None else K, X.shape[0], estimator=name
        )
        m = check_scalar(self.m, "m", estimator=name, lower=1.0, lower_inclusive=False)
        alpha = check_scalar(
            self.alpha, "alpha", estimator=name, lower=0.0, lower_inclusive=False
        )
        rng = check_generator(self.random_state)

        if self.init == "fcm":
            U, centers, _ = fcm_fit(X, K, m, rng, max_iter=self.max_iter, tol=self.tol)
        elif self.init in ("kmeans++", "random"):
            if self.init == "kmeans++":
                centers = _init_centers_kpp(X, K, rng)
            else:
                centers = X[rng.choice(X.shape[0], size=K, replace=False)].copy()
            labels = np.argmin(_euclidean_dist2(X, centers), axis=1)
            U = np.zeros((X.shape[0], K))
            U[np.arange(X.shape[0]), labels] = 1.0
        else:
            raise ValueError(
                f"{name}: init must be 'fcm', 'kmeans++' or 'random'; got "
                f"{self.init!r}."
            )

        etas = _estimate_etas(X, U, centers, m, alpha=alpha)
        if np.any(etas == 0):
            # Only true zero spread uses the data-derived global fallback.
            # If that is also zero, the typicality helper handles its limit.
            fallback = float(np.mean(_euclidean_dist2(X, centers)))
            etas = np.where(etas == 0, fallback, etas)

        T = _update_T_from_centers(X, centers, etas, m)
        obj_prev = np.inf
        trajectory = []
        for _ in range(self.max_iter):
            centers = _update_centers_from_T(X, T, m)
            T = _update_T_from_centers(X, centers, etas, m)
            obj = _objective_pcm(X, T, centers, etas, m)
            trajectory.append(obj)
            if objective_converged(obj_prev, obj, self.tol):
                break
            obj_prev = obj

        self.centers_ = centers
        self.typicalities_ = T
        self.etas_ = etas
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = len(trajectory)
        return self.typicalities_

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return _update_T_from_centers(X, self.centers_, self.etas_, self.m)
