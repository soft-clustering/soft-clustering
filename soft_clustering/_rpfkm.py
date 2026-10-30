# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_array, check_generator, check_n_clusters, check_scalar


@typechecked
class RPFKM(BaseSoftClusterer):
    r"""Robust projected fuzzy k-means (RPFKM).

    Clusters the data in a learned ``d``-dimensional linear projection
    :math:`W^\top x` while down-weighting outliers. Prototypes :math:`m_k`
    live in the projected space and distances enter through the unsquared
    norm :math:`\lVert W^\top x_i - m_k\rVert`, optimised by iteratively
    re-weighted least squares with weights
    :math:`p_{ik} = 1 / (2\lVert W^\top x_i - m_k\rVert)`. Memberships are a
    softmax of the projected distances with temperature :math:`2\gamma`, and
    the projection is updated to the leading eigenvectors of
    :math:`\beta S_t - S_w`, the total scatter minus the re-weighted
    within-cluster scatter.

    **Verification status.** The update rules follow the reference as
    implemented by the original contributors; they have not been checked
    against the paper equation by equation or against an independent
    implementation, and no objective is tracked. Treat results as those of
    an SCPP implementation in the spirit of the reference.

    Parameters
    ----------
    n_clusters : int, default=3
    d : int, default=2
        Dimension of the projection, ``1 <= d <= n_features``.
    gamma : float, default=0.1
        Membership temperature, ``> 0``; larger is softer.
    beta : float, default=1.0
        Weight of the total-scatter term in the projection update.
    max_iter : int, default=50
    random_state : int, numpy Generator or None, default=None

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    projection_ : ndarray of shape (n_features, d)
        The learned projection :math:`W`.

    References
    ----------
    X. Zhao, F. Nie, R. Wang and X. Li. Improving projected fuzzy K-means
    clustering via robust learning. *Neurocomputing*, 491:34--43, 2022.
    doi:10.1016/j.neucom.2022.03.043
    """

    # Memberships are a softmax of the projected distances (see _update_U),
    # so they are normalised over clusters by construction.
    _partition_constrained = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_clusters: int = 3,
        d: int = 2,
        gamma: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 50,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.c = n_clusters
        self.d = d
        self.gamma = gamma
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Runs the RPFKM algorithm on the given data matrix X.

        Parameters:
        - X (np.ndarray): Data matrix of shape (n_samples, n_features), as for
          every other feature-matrix estimator in SCPP. The internal updates
          below follow the paper's convention of features-by-samples, so the
          input is transposed once here.

        Returns:
        - memberships (np.ndarray): Fuzzy membership matrix, shape
          (n_samples, n_clusters), also stored as ``memberships_``. The
          learned projection is stored as ``projection_``.
        """
        name = type(self).__name__
        X = check_array(X, estimator=name)
        self.c = check_n_clusters(self.n_clusters, X.shape[0], estimator=name)
        if not 1 <= self.d <= X.shape[1]:
            raise ValueError(
                f"{name}: d must lie in [1, n_features={X.shape[1]}]; got {self.d}."
            )
        check_scalar(
            self.gamma, "gamma", estimator=name, lower=0.0, lower_inclusive=False
        )
        X = X.T  # -> (n_features, n_samples)
        W, U, p = self._initialize_variables(X)
        for _ in range(self.max_iter):
            M = self._update_M(X, W, U, p)
            p = self._update_p(X, W, M)
            U = self._update_U(X, W, M)
            W = self._update_W(X, U, p)
        # U is computed as (n_clusters, n_samples); publish it samples-first.
        self.memberships_ = U.T
        self.projection_ = W
        return self.memberships_

    def _initialize_variables(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        D, N = X.shape
        rng = check_generator(self.random_state)
        W = np.linalg.qr(rng.standard_normal((D, self.d)))[0]
        U = rng.dirichlet(np.ones(self.c), size=N).T
        p = np.ones((self.c, N))
        return W, U, p

    def _update_M(
        self, X: np.ndarray, W: np.ndarray, U: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        WX = W.T @ X
        M = np.zeros((self.d, self.c))
        for k in range(self.c):
            numerator = np.sum((p[k, :] * U[k, :]) * WX, axis=1)
            denominator = np.sum(p[k, :] * U[k, :])
            np.divide(numerator, denominator, out=M[:, k], where=denominator > 0)
        return M

    def _update_p(self, X: np.ndarray, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        WX = W.T @ X
        p = np.zeros((self.c, X.shape[1]))
        for k in range(self.c):
            diff = WX - M[:, k].reshape(-1, 1)
            norm = np.linalg.norm(diff, axis=0) + 1e-8
            p[k, :] = 1.0 / (2.0 * norm)
        return p

    def _update_U(self, X: np.ndarray, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        WX = W.T @ X
        f = np.zeros((self.c, WX.shape[1]))
        for k in range(self.c):
            diff = WX - M[:, k].reshape(-1, 1)
            f[k, :] = np.linalg.norm(diff, axis=0)
        u = -f / (2 * self.gamma)
        u = np.exp(u - u.max(axis=0, keepdims=True))
        u /= np.sum(u, axis=0, keepdims=True)
        return u

    def _update_W(self, X: np.ndarray, U: np.ndarray, p: np.ndarray) -> np.ndarray:
        D, N = X.shape
        X_mean = X @ np.ones((N, 1)) / N
        Xt = X - X_mean
        S_w = np.zeros((D, D))
        for k in range(self.c):
            s = p[k, :] * U[k, :]
            mass = np.sum(s)
            if mass == 0.0:
                continue
            x_mean_k = np.sum((s / mass) * X, axis=1, keepdims=True)
            diff = X - x_mean_k
            # (diff * s) @ diff.T, not diff @ np.diag(s) @ diff.T: the latter
            # allocates an n x n matrix per cluster (0.8 GB at n = 10,000).
            S_w += (diff * s) @ diff.T
        S_t = Xt @ Xt.T
        eigvals, eigvecs = np.linalg.eigh(self.beta * S_t - S_w)
        W = eigvecs[:, -self.d :]
        return W
