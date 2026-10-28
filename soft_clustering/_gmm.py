# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

from __future__ import annotations

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_n_features,
    check_scalar,
    data_scale,
)


def _logsumexp(a, axis=1):
    """Stable log-sum-exp."""
    m = np.max(a, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))


def _lloyd(X, means, max_iter=100, tol=1e-6):
    """Run Lloyd's algorithm from ``means`` and return the converged centroids.

    Seeding EM with k-means++ *centres* is not the same as seeding it with a
    converged k-means solution, and the difference is not cosmetic: on
    well-separated blobs at n = 200 the former lands EM in a local optimum
    with prototypes at (1.06, 1.73, 6.07) where the generating means are
    (0, 3, 6). scikit-learn's default runs k-means to convergence first, and
    ``init_params="kmeans"`` matches that.
    """
    scale = data_scale(X)
    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - means[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        updated = means.copy()
        for k in range(means.shape[0]):
            members = X[labels == k]
            if members.size:
                updated[k] = members.mean(axis=0)
        shift = np.abs(updated - means).max()
        means = updated
        if shift <= tol * scale:
            break
    return means


def _kmeanspp_init(X, K, rng):
    """k-means++ initialization for means."""
    n, d = X.shape
    means = np.empty((K, d), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    means[0] = X[idx0]
    for k in range(1, K):
        # distances to nearest current center
        d2_list = [np.sum((X - m) ** 2, axis=1) for m in means[:k]]
        d2_min = d2_list[0]
        for arr in d2_list[1:]:
            d2_min = np.minimum(d2_min, arr)
        s = d2_min.sum()
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(0, n)
        else:
            probs = d2_min / s
            idx = rng.choice(n, p=probs)
        means[k] = X[idx]
    return means


def _estimate_log_gaussian_prob_full(X, means, covariances, eps=1e-6):
    """Log N(x|mu,Sigma) for 'full' covariances."""
    n, d = X.shape
    K = means.shape[0]
    log_prob = np.empty((n, K), dtype=np.float64)
    I = np.eye(d, dtype=np.float64)
    for k in range(K):
        S = covariances[k] + eps * I
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            jitter = max(10 * eps, 1e-10 * max(float(np.trace(S)) / d, 1.0))
            L = np.linalg.cholesky(S + jitter * I)
        diff = X - means[k]
        y = np.linalg.solve(L, diff.T)
        sq = np.sum(y * y, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_prob[:, k] = -0.5 * (d * np.log(2.0 * np.pi) + log_det + sq)
    return log_prob


def _estimate_log_gaussian_prob_diag(X, means, covariances, eps=1e-6):
    """Log N(x|mu,diag(var)) for 'diag' covariances."""
    n, d = X.shape
    K = means.shape[0]
    log_prob = np.empty((n, K), dtype=np.float64)
    for k in range(K):
        var = np.maximum(covariances[k] + eps, 1e-300)
        diff = X - means[k]
        log_prob[:, k] = -0.5 * (
            np.sum(np.log(2.0 * np.pi * var)) + np.sum((diff * diff) / var, axis=1)
        )
    return log_prob


def _estimate_log_gaussian_prob_spherical(X, means, covariances, eps=1e-6):
    """Log N(x|mu, s^2 I) for 'spherical' covariances."""
    n, d = X.shape
    K = means.shape[0]
    log_prob = np.empty((n, K), dtype=np.float64)
    for k in range(K):
        var = max(covariances[k] + eps, 1e-300)
        diff = X - means[k]
        log_prob[:, k] = -0.5 * (
            d * np.log(2.0 * np.pi * var) + np.sum(diff * diff, axis=1) / var
        )
    return log_prob


def _estimate_log_gaussian_prob(X, means, covariances, covariance_type, eps=1e-6):
    if covariance_type == "full":
        return _estimate_log_gaussian_prob_full(X, means, covariances, eps)
    if covariance_type == "diag":
        return _estimate_log_gaussian_prob_diag(X, means, covariances, eps)
    if covariance_type == "spherical":
        return _estimate_log_gaussian_prob_spherical(X, means, covariances, eps)
    raise ValueError(f"Unknown covariance_type='{covariance_type}'.")


def _m_step(X, resp, covariance_type, reg_covar):
    """M-step: update weights, means, and covariances."""
    n, d = X.shape
    K = resp.shape[1]
    Nk = resp.sum(axis=0) + 1e-12
    weights = Nk / n
    means = (resp.T @ X) / Nk[:, None]

    if covariance_type == "full":
        covariances = np.empty((K, d, d), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            covariances[k] = (diff * resp[:, [k]]).T @ diff / Nk[k]
            covariances[k].flat[:: d + 1] += reg_covar
    elif covariance_type == "diag":
        covariances = np.empty((K, d), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            covariances[k] = (resp[:, [k]] * (diff * diff)).sum(axis=0) / Nk[
                k
            ] + reg_covar
    elif covariance_type == "spherical":
        covariances = np.empty((K,), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            covariances[k] = (resp[:, k] * np.sum(diff * diff, axis=1)).sum() / (
                Nk[k] * d
            ) + reg_covar
    else:
        raise ValueError(f"Unknown covariance_type='{covariance_type}'.")
    return weights, means, covariances


@typechecked
class GaussianMixtureEM(BaseSoftClusterer):
    r"""Gaussian mixture model fitted by expectation--maximisation.

    Models the data as :math:`p(x) = \sum_k \pi_k\, \mathcal{N}(x \mid
    \mu_k, \Sigma_k)` and maximises the log-likelihood with EM (Dempster et
    al., 1977). The soft output is the posterior responsibility

    .. math::

        r_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}
                      {\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)} ,

    evaluated in log space. The log-likelihood recorded in
    ``log_likelihood_trajectory_`` is non-decreasing, as EM guarantees.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of mixture components.
    covariance_type : {"full", "diag", "spherical"}, default="full"
        Covariance parameterisation.
    reg_covar : float, default=1e-6
        Non-negative ridge added to the diagonal of every covariance in the
        M-step, as in scikit-learn.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-3
        Stop when the total log-likelihood improves by at most ``tol``.
    init_params : {"kmeans", "kmeans++", "random"}, default="kmeans"
        ``"kmeans"`` runs Lloyd's algorithm from k-means++ seeds and starts EM
        from the converged centroids (scikit-learn's default);
        ``"kmeans++"`` uses the seeds directly; ``"random"`` picks random
        samples.
    random_state : int, numpy Generator or None, default=None
        Seed for the initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Posterior responsibilities; rows sum to one.
    responsibilities_ : ndarray
        Same array as ``memberships_``.
    weights_, means_, covariances_ : ndarray
        Fitted mixture parameters. ``centers_`` is ``means_``.
    lower_bound_ : float
        Total log-likelihood of the returned parameters.
    log_likelihood_trajectory_ : ndarray
        Log-likelihood at every iteration (also ``objective_trajectory_``).
    n_iter_ : int
    converged_ : bool

    References
    ----------
    A. P. Dempster, N. M. Laird and D. B. Rubin. Maximum likelihood from
    incomplete data via the EM algorithm. *Journal of the Royal Statistical
    Society, Series B*, 39(1):1--22, 1977.
    doi:10.1111/j.2517-6161.1977.tb01600.x
    """

    _membership_attrs = ("responsibilities_",)
    _centers_attrs = ("means_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-3,
        init_params: str = "kmeans",
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.init_params = init_params
        self.random_state = random_state

        # Canonical name required by the estimator protocol (see _base.py).

    def fit_predict(self, X, y=None, K: int | None = None):
        """Fit the model with EM and return the responsibilities.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
        y : ignored
            Accepted for scikit-learn compatibility.
        K : int, optional
            Deprecated fit-time component count; prefer ``n_clusters``.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
        """
        name = type(self).__name__
        X = check_array(X, estimator=name)
        n, d = X.shape
        K = check_n_clusters(self.n_clusters if K is None else K, n, estimator=name)
        reg = check_scalar(self.reg_covar, "reg_covar", estimator=name, lower=0.0)
        if self.covariance_type not in ("full", "diag", "spherical"):
            raise ValueError(
                f"{name}: covariance_type must be 'full', 'diag' or 'spherical'; "
                f"got {self.covariance_type!r}."
            )

        rng = check_generator(self.random_state)

        if self.init_params in ("kmeans", "kmeans++"):
            means = _kmeanspp_init(X, K, rng)
            if self.init_params == "kmeans":
                means = _lloyd(X, means)
        elif self.init_params == "random":
            means = X[rng.choice(n, size=K, replace=False)].copy()
        else:
            raise ValueError(
                f"{name}: init_params must be 'kmeans', 'kmeans++' or 'random'; "
                f"got {self.init_params!r}."
            )

        weights = np.full(K, 1.0 / K, dtype=np.float64)

        # Initial covariances from the global covariance.
        Xc = X - np.mean(X, axis=0, keepdims=True)
        S = (Xc.T @ Xc) / max(n - 1, 1)
        if self.covariance_type == "full":
            covariances = np.array(
                [S + reg * np.eye(d) for _ in range(K)], dtype=np.float64
            )
        elif self.covariance_type == "diag":
            covariances = np.tile(np.diag(S) + reg, (K, 1))
        else:
            covariances = np.full(K, np.mean(np.diag(S)) + reg, dtype=np.float64)

        ll_old = -np.inf
        ll_trace = []
        converged = False
        for _ in range(self.max_iter):
            log_resp, ll = _e_step(X, weights, means, covariances, self.covariance_type)
            ll_trace.append(ll)
            resp = np.exp(log_resp)
            weights, means, covariances = _m_step(X, resp, self.covariance_type, reg)
            if abs(ll - ll_old) <= self.tol:
                converged = True
                break
            ll_old = ll

        # Responsibilities consistent with the returned parameters.
        log_resp, ll = _e_step(X, weights, means, covariances, self.covariance_type)
        resp = np.exp(log_resp)

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.responsibilities_ = resp
        self.lower_bound_ = ll
        self.log_likelihood_trajectory_ = np.array(ll_trace, dtype=np.float64)
        self.objective_trajectory_ = self.log_likelihood_trajectory_
        self.n_iter_ = len(ll_trace)
        self.converged_ = converged
        return resp

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.means_.shape[1], estimator=type(self).__name__)
        log_resp, _ = _e_step(
            X, self.weights_, self.means_, self.covariances_, self.covariance_type
        )
        return np.exp(log_resp)

    def score_samples(self, X) -> np.ndarray:
        """Log-density of each sample under the fitted mixture."""
        self._check_fitted()
        X = check_array(X, estimator=type(self).__name__)
        log_prob = _estimate_log_gaussian_prob(
            X, self.means_, self.covariances_, self.covariance_type, eps=0.0
        ) + np.log(np.maximum(self.weights_, 1e-300))
        return _logsumexp(log_prob, axis=1).ravel()


def _e_step(X, weights, means, covariances, covariance_type):
    """Return ``(log_responsibilities, total_log_likelihood)``.

    ``reg_covar`` has already been added to the covariances by the M-step, so
    they are used as they are. Earlier releases added it a second time here.
    """
    log_prob = _estimate_log_gaussian_prob(
        X, means, covariances, covariance_type, eps=0.0
    )
    log_prob += np.log(np.maximum(weights, 1e-300))
    log_norm = _logsumexp(log_prob, axis=1)
    return log_prob - log_norm, float(np.sum(log_norm))
