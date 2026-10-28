# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Beta--Gaussian mixture model for two aligned data views (Dai et al., 2008)."""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, xlog1py, xlogy
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._gmm import _kmeanspp_init, _lloyd, _logsumexp
from ._validation import check_array, check_generator, check_n_clusters


def _as_column(values, name, estimator):
    arr = check_array(
        np.asarray(values, dtype=float).reshape(-1, 1), estimator=estimator, name=name
    )
    return arr.ravel()


@typechecked
class BGMM(BaseSoftClusterer):
    r"""Beta--Gaussian mixture model over two aligned views.

    Each sample :math:`i` has a real-valued observation :math:`g_i` and an
    observation :math:`b_i \in (0, 1)` (for example an expression level and a
    probability or proportion from a second source). Given the component, the
    two views are modelled as independent:

    .. math::

        p(g_i, b_i) = \sum_k \pi_k\, \mathcal{N}(g_i \mid \mu_k, \sigma_k^2)\,
                      \mathrm{Beta}(b_i \mid a_k, c_k),

    fitted by EM. The soft output is the posterior responsibility of each
    component.

    The Gaussian parameters are updated by maximum likelihood. The Beta
    parameters are updated by the method of moments from the
    responsibility-weighted mean and variance, a common approximation that
    does not guarantee a monotone likelihood; ``log_likelihood_trajectory_``
    records the observed-data log-likelihood so this can be checked.

    EM is started from a k-means partition of the two standardised views,
    which breaks the symmetry between components. Earlier releases started
    every component from identical parameters with uniform responsibilities,
    a fixed point of EM: the fit never moved and every membership was
    exactly ``1/K``.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of components.
    max_iter : int, default=100
    tol : float, default=1e-5
        Stop when the log-likelihood changes by less than ``tol``.
    random_state : int, numpy Generator or None, default=None
        Seed for the k-means initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    weights : ndarray of shape (n_clusters,)
    gaussian_params : list of (mean, std)
    beta_params : list of (a, c)
    log_likelihood_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    X. Dai, H. Lahdesmaki and O. Yli-Harja. BGMM: a Beta-Gaussian mixture
    model for clustering genes with multiple data sources. In *Proc. 5th
    International Workshop on Computational Systems Biology (WCSB 2008)*,
    pp. 25--28, 2008.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    # Fitted attributes, set by ``fit`` (not before).
    weights: np.ndarray
    gaussian_params: list[tuple[float, float]]
    beta_params: list[tuple[float, float]]
    resp: np.ndarray

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_components = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _log_joint(g, b, weights, mu, sd, a, c):
        log_gauss = (
            -0.5 * np.log(2 * np.pi * sd**2)[None, :]
            - 0.5 * ((g[:, None] - mu[None, :]) / sd[None, :]) ** 2
        )
        log_beta = (
            xlogy(a[None, :] - 1.0, b[:, None])
            + xlog1py(c[None, :] - 1.0, -b[:, None])
            - betaln(a, c)[None, :]
        )
        return np.log(np.maximum(weights, 1e-300))[None, :] + log_gauss + log_beta

    @staticmethod
    def _m_step(g, b, R):
        Nk = np.maximum(R.sum(axis=0), 1e-10)
        weights = Nk / R.shape[0]
        mu = (R * g[:, None]).sum(axis=0) / Nk
        sd = np.sqrt((R * (g[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk)
        sd = np.maximum(sd, 1e-3 * max(float(np.std(g)), 1e-12))
        mean = (R * b[:, None]).sum(axis=0) / Nk
        var = (R * (b[:, None] - mean[None, :]) ** 2).sum(axis=0) / Nk
        mean = np.clip(mean, 1e-3, 1 - 1e-3)
        var = np.clip(var, 1e-6, mean * (1 - mean) * (1 - 1e-6))
        common = mean * (1 - mean) / var - 1.0
        a = np.maximum(mean * common, 1e-2)
        c = np.maximum((1 - mean) * common, 1e-2)
        return weights, mu, sd, a, c

    def fit(self, Xg, Xb) -> BGMM:
        """Fit on two aligned views.

        Parameters
        ----------
        Xg : array-like of shape (n_samples,)
            The Gaussian view (any real values).
        Xb : array-like of shape (n_samples,)
            The Beta view; every value must lie in the open interval (0, 1).
        """
        name = type(self).__name__
        g = _as_column(Xg, "Xg", name)
        b = _as_column(Xb, "Xb", name)
        if g.shape != b.shape:
            raise ValueError(
                f"{name}: Xg and Xb must have the same length; got {g.size} and {b.size}."
            )
        if np.any(b <= 0) or np.any(b >= 1):
            raise ValueError(
                f"{name}: every value of Xb must lie in the open interval (0, 1); "
                f"got values in [{b.min():.6g}, {b.max():.6g}]."
            )
        n = g.size
        K = check_n_clusters(self.n_clusters, n, estimator=name)
        rng = check_generator(self.random_state)

        Z = np.column_stack([g, b])
        Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
        centers = _lloyd(Z, _kmeanspp_init(Z, K, rng))
        labels = np.argmin(
            ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        R = np.full((n, K), 1e-3)
        R[np.arange(n), labels] = 1.0
        R /= R.sum(axis=1, keepdims=True)

        trace = []
        prev = -np.inf
        for _ in range(self.max_iter):
            params = self._m_step(g, b, R)
            log_joint = self._log_joint(g, b, *params)
            log_norm = _logsumexp(log_joint, axis=1)
            R = np.exp(log_joint - log_norm)
            ll = float(log_norm.sum())
            trace.append(ll)
            if abs(ll - prev) < self.tol:
                break
            prev = ll

        weights, mu, sd, a, c = params
        self.weights = weights
        self.gaussian_params = [(float(m_), float(s_)) for m_, s_ in zip(mu, sd)]
        self.beta_params = [(float(a_), float(c_)) for a_, c_ in zip(a, c)]
        self.resp = R
        self.memberships_ = R
        self.log_likelihood_trajectory_ = np.array(trace)
        self.n_iter_ = len(trace)
        return self
