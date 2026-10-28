# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Gath--Geva fuzzy maximum-likelihood estimation clustering.

Reference
---------
I. Gath and A. B. Geva. *Fuzzy clustering for the estimation of the parameters
of the components of mixtures of normal distributions.* Pattern Recognition
Letters, 9(2):77--86, 1989.

The method replaces the Euclidean distance of fuzzy c-means with an
exponential distance derived from the assumption that each cluster is a normal
component with its own covariance and prior:

.. math::

    d^2_{ik} \\;=\\; \\frac{\\sqrt{\\det F_i}}{P_i}
        \\exp\\!\\Big(\\tfrac12 (x_k - v_i)^\\top F_i^{-1} (x_k - v_i)\\Big),

with fuzzy covariance :math:`F_i = \\sum_k u_{ik}^m (x_k - v_i)(x_k -
v_i)^\\top / \\sum_k u_{ik}^m` and a-priori probability :math:`P_i =
\\frac1n \\sum_k u_{ik}`. Memberships follow the usual fuzzy c-means rule
applied to these distances. The prior uses the memberships themselves, not
their powers, so that the :math:`P_i` sum to one; earlier releases used
:math:`u_{ik}^m`, which does not define a probability distribution.

Implementation note
-------------------
The exponential distance overflows in double precision as soon as the
Mahalanobis term exceeds roughly 1400, which happens routinely on outliers.
The update is therefore evaluated in log space throughout:
:math:`\\log d^2_{ik} = \\tfrac12 \\log\\det F_i - \\log P_i + \\tfrac12
(x_k-v_i)^\\top F_i^{-1}(x_k-v_i)`, and

.. math::

    u_{ik} \\;=\\; \\frac{(d^2_{ik})^{-1/(m-1)}}
                        {\\sum_j (d^2_{jk})^{-1/(m-1)}}
           \\;=\\; \\mathrm{softmax}_i\\!\\Big(\\frac{-\\log d^2_{ik}}{m-1}\\Big),

which is the same quantity computed without ever forming the exponential.
"""

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
)


@typechecked
class GathGeva(BaseSoftClusterer):
    """Gath--Geva (fuzzy maximum-likelihood estimation) clustering."""

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-5,
        reg_covar: float = 1e-6,
        init: str = "fcm",
        init_iter: int = 20,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        m : float
            Fuzzifier, ``m > 1``.
        max_iter : int
            Maximum number of alternating-optimisation sweeps.
        tol : float
            Convergence threshold on the maximum membership change.
        reg_covar : float
            Ridge added to the diagonal of every fuzzy covariance, so that a
            cluster collapsing onto fewer points than dimensions stays
            invertible.
        init : {"fcm", "random"}
            ``"fcm"`` seeds the memberships with ``init_iter`` fuzzy c-means
            sweeps, which is what Gath and Geva prescribe: the exponential
            distance has many poor local optima and the method is defined as a
            refinement of a fuzzy c-means partition. ``"random"`` starts from
            a random partition and is provided for ablation.
        init_iter : int
            Number of fuzzy c-means sweeps used by ``init="fcm"``.
        random_state : int or None
            Seed for the membership initialisation.
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init = init
        self.init_iter = init_iter
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_sq_distances(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Return ``log d^2`` of shape ``(n_samples, n_clusters)``."""
        n_samples, n_features = X.shape
        U_m = U**self.m
        # A cluster can lose all its mass; flooring the weight keeps the
        # prototype finite instead of propagating NaN through the sweep.
        weights = np.maximum(U_m.sum(axis=0), np.finfo(float).tiny)

        # Fuzzy means and covariances.
        centers = (U_m.T @ X) / weights[:, None]
        diff = X[:, None, :] - centers[None, :, :]  # (n, K, d)
        cov = np.einsum("nk,nkd,nke->kde", U_m, diff, diff, optimize=True)
        cov /= weights[:, None, None]

        # Cholesky is both the stable inverse and the stable log-determinant,
        # but a cluster that has collapsed onto a lower-dimensional subspace
        # gives a singular covariance. Grow the ridge until it factorises;
        # the scale is set by the trace so the ridge stays relative.
        eye = np.eye(n_features)
        scale = max(float(np.mean(np.abs(np.trace(cov, axis1=1, axis2=2)))), 1.0)
        ridge = self.reg_covar
        for _ in range(12):
            try:
                chol = np.linalg.cholesky(cov + eye * ridge * scale)
                break
            except np.linalg.LinAlgError:
                ridge = ridge * 10 if ridge > 0 else 1e-10
        else:  # pragma: no cover - a 1e6 relative ridge is already diagonal
            chol = np.linalg.cholesky(cov + eye * scale)
        cov = cov + eye * ridge * scale
        # solve L z = diff^T  per cluster; maha = ||z||^2
        z = np.linalg.solve(chol[None, :, :, :], diff[:, :, :, None])
        maha = np.einsum("nkdi,nkdi->nk", z, z, optimize=True)
        log_det = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)

        priors = U.sum(axis=0) / n_samples
        priors = np.maximum(priors, np.finfo(float).tiny)

        self.centers_ = centers
        self.covariances_ = cov
        self.priors_ = priors

        return 0.5 * log_det[None, :] - np.log(priors)[None, :] + 0.5 * maha

    def _fcm_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Fuzzy c-means memberships, used to seed the exponential distance."""
        n_samples = X.shape[0]
        # k-means++ style seeding keeps the initial prototypes apart.
        seeds = [X[rng.integers(n_samples)]]
        for _ in range(1, self.n_clusters):
            d2 = np.min(
                ((X[:, None, :] - np.asarray(seeds)[None, :, :]) ** 2).sum(axis=2),
                axis=1,
            )
            total = d2.sum()
            probabilities = d2 / total if total > 0 else None
            seeds.append(X[rng.choice(n_samples, p=probabilities)])
        centers = np.asarray(seeds)

        exponent = 2.0 / (self.m - 1.0)
        U = np.empty((n_samples, self.n_clusters))  # init_iter >= 1 (fit)
        for _ in range(self.init_iter):
            d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            # Evaluate the ratio rule in log space and subtract the per-sample
            # minimum. That factor cancels between numerator and denominator
            # and caps every exponent at zero, so a point coincident with a
            # prototype underflows to a clean one-hot row instead of
            # overflowing to inf/nan.
            log_d = np.log(np.maximum(d, np.finfo(float).tiny))
            scores = -exponent * (log_d - log_d.min(axis=1, keepdims=True))
            np.exp(scores, out=scores)
            U = scores / scores.sum(axis=1, keepdims=True)
            U_m = U**self.m
            weights = np.maximum(U_m.sum(axis=0), np.finfo(float).tiny)
            centers = (U_m.T @ X) / weights[:, None]
        return U

    # ------------------------------------------------------------------
    # Estimator interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> GathGeva:
        """Fit the model on a feature matrix ``X`` of shape ``(n_samples, n_features)``."""
        X = check_array(X, estimator=type(self).__name__)
        n_samples = X.shape[0]
        name = type(self).__name__
        check_n_clusters(self.n_clusters, n_samples, estimator=name)
        check_scalar(self.m, "m", estimator=name, lower=1.0, lower_inclusive=False)
        check_scalar(self.reg_covar, "reg_covar", estimator=name, lower=0.0)
        if self.max_iter < 1:
            raise ValueError(f"{name}: max_iter must be >= 1; got {self.max_iter}.")
        if self.init_iter < 1:
            raise ValueError(f"{name}: init_iter must be >= 1; got {self.init_iter}.")
        if self.init not in ("fcm", "random"):
            raise ValueError(
                f"{name}: init must be 'fcm' or 'random'; got {self.init!r}."
            )

        rng = check_generator(self.random_state)
        if self.init == "fcm":
            U = self._fcm_init(X, rng)
        else:
            U = rng.random((n_samples, self.n_clusters)) + 1e-3
            U /= U.sum(axis=1, keepdims=True)

        exponent = 1.0 / (self.m - 1.0)
        self.n_iter_ = self.max_iter

        for iteration in range(self.max_iter):
            U_prev = U
            log_d2 = self._log_sq_distances(X, U_prev)

            # u_ik = softmax_i(-log d^2_ik / (m - 1)), evaluated stably.
            scores = -log_d2 * exponent
            scores -= scores.max(axis=1, keepdims=True)
            np.exp(scores, out=scores)
            U = scores / scores.sum(axis=1, keepdims=True)

            if np.abs(U - U_prev).max() < self.tol:
                self.n_iter_ = iteration + 1
                break

        # Publish parameters and memberships that correspond to each other:
        # one last parameter refresh from U, then the memberships they imply.
        self._log_sq_distances(X, U)
        self.memberships_ = self._memberships_from_params(X)
        return self

    def _memberships_from_params(self, X: np.ndarray) -> np.ndarray:
        """Membership rule for the stored centres, covariances and priors."""
        chol = np.linalg.cholesky(self.covariances_)
        diff = X[:, None, :] - self.centers_[None, :, :]
        z = np.linalg.solve(chol[None, :, :, :], diff[:, :, :, None])
        maha = np.einsum("nkdi,nkdi->nk", z, z, optimize=True)
        log_det = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        log_d2 = 0.5 * log_det[None, :] - np.log(self.priors_)[None, :] + 0.5 * maha
        scores = -log_d2 / (self.m - 1.0)
        scores -= scores.max(axis=1, keepdims=True)
        np.exp(scores, out=scores)
        return scores / scores.sum(axis=1, keepdims=True)

    def _predict_memberships(self, X) -> np.ndarray:
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return self._memberships_from_params(X)
