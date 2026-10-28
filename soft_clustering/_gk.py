# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

from __future__ import annotations

import numpy as np
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


def _kmeanspp_init(X, K, rng):
    """k-means++ style seeding from data points."""
    n, d = X.shape
    centers = np.empty((K, d), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    for k in range(1, K):
        d2_list = [np.sum((X - c) ** 2, axis=1) for c in centers[:k]]
        d2_min = d2_list[0]
        for arr in d2_list[1:]:
            d2_min = np.minimum(d2_min, arr)
        s = float(d2_min.sum())
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(0, n)
        else:
            probs = d2_min / s
            idx = rng.choice(n, p=probs)
        centers[k] = X[idx]
    return centers


def _compute_covariances(X, U, centers, m, reg_covar=1e-6):
    """Fuzzy covariance per cluster."""
    n, d = X.shape
    K = centers.shape[0]
    C = np.empty((K, d, d), dtype=np.float64)
    Um = U**m
    for k in range(K):
        w = Um[:, k : k + 1]
        diff = X - centers[k]
        Nk = float(np.sum(w)) + 1e-12
        S = (diff * w).T @ diff / Nk
        S.flat[:: d + 1] += reg_covar
        C[k] = S
    return C


def _estimate_gk_dist2(X, centers, covariances):
    """GK distance: det(C_k)^{1/d} * (x - c_k)^T C_k^{-1} (x - c_k)."""
    n, d = X.shape
    K = centers.shape[0]
    D2 = np.empty((n, K), dtype=np.float64)
    I = None
    for k in range(K):
        S = covariances[k]
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            if I is None:
                I = np.eye(S.shape[0], dtype=np.float64)
            S = S + 10.0 * np.finfo(np.float64).eps * I
            L = np.linalg.cholesky(S)
        diff = X - centers[k]
        y = np.linalg.solve(L, diff.T)
        sq = np.sum(y * y, axis=0)  # Mahalanobis^2
        log_det = 2.0 * np.sum(np.log(np.diag(L)))  # log|C_k|
        scale = np.exp(log_det / d)  # det(C_k)^{1/d}
        D2[:, k] = scale * sq
    np.maximum(D2, 1e-18, out=D2)
    return D2


def _update_U_from_d2(d2, m, eps=1e-12):
    """Update memberships given GK distances.

    The ratio rule is delegated to the shared, overflow-free helper. Written
    directly as ``d2 ** -p / sum_j d2 ** -p`` it overflows for a fuzzifier
    close to one -- ``p = 1/(m - 1)`` is 100 at ``m = 1.01`` -- and the
    normalisation turns the resulting ``inf`` into an all-``NaN`` row, which
    then poisons the fuzzy covariances on the next sweep.

    The floor afterwards is specific to GK: a membership of exactly zero for
    every sample in a cluster leaves that cluster's scatter matrix with no
    weight at all, so the memberships are kept strictly positive.
    """
    U = ratio_memberships(d2, 1.0 / (m - 1.0))
    U = np.maximum(U, eps)
    U /= np.sum(U, axis=1, keepdims=True)
    return U


def _objective(U, d2, m):
    return float(np.sum((U**m) * d2))


@typechecked
class GustafsonKessel(BaseSoftClusterer):
    r"""Gustafson--Kessel fuzzy clustering (GK).

    Fuzzy c-means with an adaptive, cluster-specific Mahalanobis norm. Each
    cluster carries a fuzzy covariance

    .. math::

        F_k = \frac{\sum_i u_{ik}^m (x_i - v_k)(x_i - v_k)^\top}{\sum_i u_{ik}^m},

    and distances are measured in the volume-normalised norm
    :math:`A_k = \det(F_k)^{1/d} F_k^{-1}` (cluster volume
    :math:`\rho_k = 1`), so clusters may take different ellipsoidal shapes
    but not different sizes. Memberships and prototypes are then updated as
    in fuzzy c-means.

    A small ridge ``reg_covar`` is added to every covariance, so a cluster
    collapsing onto fewer points than dimensions stays invertible; this is
    the usual practical safeguard (Babuska et al., 2002) and biases the
    norm negligibly for well-conditioned clusters.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    m : float, default=2.0
        Fuzzifier, ``m > 1``.
    max_iter : int, default=300
        Maximum number of sweeps.
    tol : float, default=1e-5
        Stop when the objective changes by at most ``tol`` relative to its
        magnitude.
    init : {"kmeans++", "random"}, default="kmeans++"
        Prototype initialisation.
    reg_covar : float, default=1e-6
        Ridge added to the diagonal of every fuzzy covariance.
    random_state : int, numpy Generator or None, default=None
        Seed for the initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    centers_ : ndarray of shape (n_clusters, n_features)
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariance matrices :math:`F_k`.
    metrics_A_ : ndarray of shape (n_clusters, n_features, n_features)
        Norm-inducing matrices :math:`A_k`, each with unit determinant.
    objective_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    D. E. Gustafson and W. C. Kessel. Fuzzy clustering with a fuzzy
    covariance matrix. In *Proc. IEEE Conference on Decision and Control*,
    pp. 761--766, 1979. doi:10.1109/CDC.1978.268028

    R. Babuska, P. J. van der Veen and U. Kaymak. Improved covariance
    estimation for Gustafson--Kessel clustering. In *Proc. IEEE International
    Conference on Fuzzy Systems*, pp. 1081--1085, 2002.
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
        reg_covar: float = 1e-6,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit_predict(self, X, y=None, K: int | None = None):
        """Fit the model and return fuzzy memberships.

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
        n, d = X.shape
        K = check_n_clusters(self.n_clusters if K is None else K, n, estimator=name)
        m = check_scalar(self.m, "m", estimator=name, lower=1.0, lower_inclusive=False)
        check_scalar(self.reg_covar, "reg_covar", estimator=name, lower=0.0)

        rng = check_generator(self.random_state)

        if self.init == "kmeans++":
            centers = _kmeanspp_init(X, K, rng)
            U = np.full((n, K), 1.0 / K, dtype=np.float64)
        elif self.init == "random":
            U = rng.random((n, K))
            U = U / (np.sum(U, axis=1, keepdims=True) + 1e-12)
            Um = U**m
            denom = np.sum(Um, axis=0, keepdims=True).T
            centers = (Um.T @ X) / (denom + 1e-12)
        else:
            raise ValueError(
                f"{name}: init must be 'kmeans++' or 'random'; got {self.init!r}."
            )

        obj_prev = np.inf
        traj = []
        for _ in range(self.max_iter):
            C = _compute_covariances(X, U, centers, m, reg_covar=self.reg_covar)
            d2 = _estimate_gk_dist2(X, centers, C)
            U = _update_U_from_d2(d2, m)
            Um = U**m
            denom = np.sum(Um, axis=0, keepdims=True).T
            centers = (Um.T @ X) / (denom + 1e-12)
            obj = _objective(U, d2, m)
            traj.append(obj)
            if objective_converged(obj_prev, obj, self.tol):
                break
            obj_prev = obj

        # Finish with a covariance and membership update against the final
        # prototypes, so that the returned (U, V, F) are mutually consistent
        # and predict_proba on the training data reproduces memberships_.
        C = _compute_covariances(X, U, centers, m, reg_covar=self.reg_covar)
        U = _update_U_from_d2(_estimate_gk_dist2(X, centers, C), m)
        self.centers_ = centers
        self.memberships_ = U
        self.covariances_ = C
        self.metrics_A_ = _norm_matrices(C)
        self.objective_trajectory_ = np.array(traj, dtype=np.float64)
        self.n_iter_ = len(traj)
        return self.memberships_

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        d2 = _estimate_gk_dist2(X, self.centers_, self.covariances_)
        return _update_U_from_d2(d2, self.m)


def _norm_matrices(C):
    """``A_k = det(C_k)^{1/d} C_k^{-1}``, so that ``det(A_k) = 1``."""
    K, d = C.shape[0], C.shape[1]
    A = np.empty_like(C)
    for k in range(K):
        S = C[k]
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S = S + 10.0 * np.finfo(np.float64).eps * np.eye(d, dtype=np.float64)
            L = np.linalg.cholesky(S)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        A[k] = np.exp(log_det / d) * np.linalg.inv(S)
    return A
