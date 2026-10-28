# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Adaptive fuzzy c-means with graph embedding (Chen et al., 2024)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from typeguard import typechecked

from ._afcmSimple import adaptive_fcm_step
from ._base import BaseSoftClusterer
from ._fcm import _euclidean_dist2, _init_centers_kpp
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_scalar,
    objective_converged,
)


def _normalized_knn_laplacian(X: np.ndarray, n_neighbors: int, sigma: float | None):
    """``L_hat = I - D^{-1/2} W D^{-1/2}`` of a symmetrised Gaussian kNN graph."""
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    dist, ind = dist[:, 1:], ind[:, 1:]  # drop the sample itself
    if sigma is None:
        positive = dist[dist > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, ind.ravel()] = np.exp(-(dist.ravel() ** 2) / (2.0 * sigma**2))
    W = np.maximum(W, W.T)
    degree = W.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-300)), 0.0)
    L_hat = np.eye(n) - inv_sqrt[:, None] * W * inv_sqrt[None, :]
    return 0.5 * (L_hat + L_hat.T), sigma


@typechecked
class AFCM(BaseSoftClusterer):
    r"""Adaptive fuzzy c-means with graph embedding (AFCM).

    Clusters a learned low-dimensional embedding :math:`\tilde X \in
    \mathbb{R}^{K \times n}` of the data instead of the data itself, and
    learns the fuzziness from the data (Chen et al., 2024):

    .. math::

        \min_{U, V, \gamma, \tilde X}\;
            \gamma \sum_{i,j} u_{ij}\lVert \tilde x_i - v_j\rVert^2
            + \lambda\, \mathrm{Tr}(\tilde X \hat L \tilde X^\top)
            + \sum_{i,j} u_{ij}\log u_{ij} - \frac{n K}{2}\log\gamma
        \quad\text{s.t. } \tilde X \tilde X^\top = I,\; \sum_j u_{ij} = 1,

    where :math:`\hat L = I - D^{-1/2} W D^{-1/2}` is the normalised
    Laplacian of a Gaussian k-nearest-neighbour graph. Given the embedding,
    :math:`U, V, \gamma` have the closed forms of :class:`AFCMSimple`; given
    those, the embedding is spanned by the :math:`K` eigenvectors with the
    smallest eigenvalues of :math:`\gamma(I - U B U^\top) + \lambda \hat L`,
    with :math:`B = \mathrm{diag}(1/\sum_i u_{ij})`. Each block update
    decreases the objective, so ``objective_trajectory_`` is non-increasing.

    The graph term lets the method separate non-convex clusters that a
    prototype method in the input space cannot.

    **Cost.** Every sweep solves a dense ``n x n`` eigenproblem, so time is
    :math:`O(n^3)` per sweep and memory :math:`O(n^2)`; the method is meant
    for data sets of up to a few thousand samples.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters, which is also the embedding dimension.
    lambda_ : float, default=1.0
        Weight :math:`\lambda \ge 0` of the graph-embedding regulariser.
    n_neighbors : int, default=5
        Neighbours per sample in the kNN graph.
    sigma : float or None, default=None
        Bandwidth of the Gaussian edge weights; ``None`` uses the median
        distance to the ``n_neighbors`` nearest neighbours.
    max_iter : int, default=100
    tol : float, default=1e-5
        Stop when the objective changes by at most ``tol`` relative to its
        magnitude.
    random_state : int, numpy Generator or None, default=None
        Seed for the k-means++ initialisation in the initial embedding.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    embedding_ : ndarray of shape (n_samples, n_clusters)
        The learned embedding :math:`\tilde X^\top`.
    centers_ : None
        Prototypes live in the embedding, not in the input space; they are
        in ``embedding_centers_``.
    embedding_centers_ : ndarray of shape (n_clusters, n_clusters)
    gamma_ : float
    objective_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    Q. Chen, W. Yu, F. Nie and X. Li. Adaptive fuzzy c-means with graph
    embedding. arXiv:2405.13427, 2024.

    A. Y. Ng, M. I. Jordan and Y. Weiss. On spectral clustering: analysis and
    an algorithm. In *Advances in Neural Information Processing Systems 14*,
    2002. (Row normalisation of the initial spectral embedding.)

    Notes
    -----
    Earlier releases used the fuzzy c-means ratio rule with a fixed
    fuzzifier ``m`` and an unweighted, unnormalised kNN Laplacian, which is
    not the reference model; ``m`` has been removed.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_clusters: int = 3,
        lambda_: float = 1.0,
        n_neighbors: int = 5,
        sigma: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.lambda_ = lambda_
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X) -> AFCM:
        """Fit on a feature matrix of shape ``(n_samples, n_features)``."""
        name = type(self).__name__
        X = check_array(X, estimator=name, min_samples=2)
        n = X.shape[0]
        K = check_n_clusters(self.n_clusters, n, estimator=name)
        lam = check_scalar(self.lambda_, "lambda_", estimator=name, lower=0.0)
        if self.n_neighbors < 1:
            raise ValueError(
                f"{name}: n_neighbors must be >= 1; got {self.n_neighbors}."
            )
        if self.sigma is not None:
            check_scalar(
                self.sigma, "sigma", estimator=name, lower=0.0, lower_inclusive=False
            )

        L_hat, self.sigma_ = _normalized_knn_laplacian(X, self.n_neighbors, self.sigma)

        # Initial embedding: the spectral embedding of the graph alone,
        # rescaled so that the adaptive precision starts at a sensible value.
        _, Z = eigh(L_hat, subset_by_index=(0, K - 1))
        # The initial partition is taken from the row-normalised embedding,
        # as in Ng, Jordan and Weiss (2002): eigenvectors of the normalised
        # Laplacian scale each row with the square root of its degree, so a
        # cluster lies on a ray from the origin and a Euclidean prototype
        # initialisation would cut rays apart.
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)
        V = _init_centers_kpp(Zn, K, check_generator(self.random_state))
        for _ in range(20):  # a few Lloyd steps on the normalised rows
            labels = np.argmin(_euclidean_dist2(Zn, V), axis=1)
            V = np.vstack(
                [
                    Zn[labels == k].mean(axis=0) if np.any(labels == k) else V[k]
                    for k in range(K)
                ]
            )
        labels = np.argmin(_euclidean_dist2(Zn, V), axis=1)
        U = np.zeros((n, K))
        U[np.arange(n), labels] = 1.0

        trajectory = []
        J_prev = np.inf
        for _ in range(self.max_iter):
            U, V, gamma, J_fuzzy = adaptive_fcm_step(Z, U)
            weights = np.maximum(U.sum(axis=0), np.finfo(float).tiny)
            P = (U / weights[None, :]) @ U.T  # U B U^T
            M = gamma * (np.eye(n) - P) + lam * L_hat
            _, Z = eigh(0.5 * (M + M.T), subset_by_index=(0, K - 1))
            # Objective at (U, gamma, Z) with the prototypes at their optimum,
            # sum_ij u_ij ||z_i - v_j||^2 = Tr(Z^T Z) - Tr(Z^T P Z) and
            # Tr(Z^T Z) = K for orthonormal columns. This is the quantity the
            # eigen-step minimises, evaluated after it.
            tiny = np.finfo(float).tiny
            J = float(
                gamma * (K - np.trace(Z.T @ P @ Z))
                + lam * np.trace(Z.T @ L_hat @ Z)
                + np.sum(U * np.log(np.maximum(U, tiny)))
                - n * (K / 2.0) * np.log(gamma)
            )
            del J_fuzzy
            trajectory.append(J)
            if objective_converged(J_prev, J, self.tol):
                break
            J_prev = J

        # Memberships, prototypes and precision for the final embedding.
        U, V, gamma, _ = adaptive_fcm_step(Z, U)
        self.embedding_ = Z
        self.embedding_centers_ = V
        self.gamma_ = gamma
        self.memberships_ = U
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = len(trajectory)
        return self
