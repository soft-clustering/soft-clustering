# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Adaptive fuzzy c-means without graph embedding (Chen et al., 2024)."""

from __future__ import annotations

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._fcm import _euclidean_dist2, _init_centers_kpp
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_n_features,
    objective_converged,
)


def adaptive_fcm_step(Z: np.ndarray, U: np.ndarray):
    """One sweep of the adaptive model on data ``Z``; returns ``(U, V, gamma, J)``.

    Implements the closed-form updates of Chen et al. (2024): prototypes are
    membership-weighted means, the precision is
    ``gamma = d n / (2 sum_ij u_ij ||z_i - v_j||^2)`` and memberships are a
    softmax of ``-gamma ||z_i - v_j||^2``.
    """
    n, d = Z.shape
    weights = np.maximum(U.sum(axis=0), np.finfo(float).tiny)
    V = (U.T @ Z) / weights[:, None]
    d2 = _euclidean_dist2(Z, V)
    spread = float(np.sum(U * d2))
    # When every cluster collapses onto a single point the within-cluster
    # spread vanishes, gamma diverges and the objective is unbounded below.
    # That is a degenerate optimum of the model, not a numerical accident;
    # gamma is capped relative to the total spread of Z so that the
    # memberships stay defined (and crisp) instead of overflowing.
    total = float(np.sum((Z - Z.mean(axis=0)) ** 2))
    floor = max(1e-10 * total, np.finfo(float).tiny * 1e10)
    gamma = d * n / (2.0 * max(spread, floor))
    scores = -gamma * d2
    scores -= scores.max(axis=1, keepdims=True)
    U_new = np.exp(scores)
    U_new /= U_new.sum(axis=1, keepdims=True)
    tiny = np.finfo(float).tiny
    J = float(
        gamma * np.sum(U_new * d2)
        + np.sum(U_new * np.log(np.maximum(U_new, tiny)))
        - n * (d / 2.0) * np.log(gamma)
    )
    return U_new, V, gamma, J


@typechecked
class AFCMSimple(BaseSoftClusterer):
    r"""Adaptive fuzzy c-means without graph embedding.

    The parameter-free model of Chen et al. (2024, Sec. 3), obtained from
    :class:`AFCM` by removing the graph-embedding regulariser:

    .. math::

        \min_{U, V, \gamma}\; \sum_{i,j} \gamma\, u_{ij} \lVert x_i - v_j\rVert^2
            + \sum_{i,j} u_{ij}\log u_{ij} - \frac{n d}{2}\log\gamma ,
        \qquad \sum_j u_{ij} = 1 .

    The fuzziness is not a user hyperparameter: :math:`\gamma` is learned,
    with closed-form updates

    .. math::

        u_{ij} = \frac{e^{-\gamma\lVert x_i - v_j\rVert^2}}
                      {\sum_l e^{-\gamma\lVert x_i - v_l\rVert^2}},\quad
        v_j = \frac{\sum_i u_{ij} x_i}{\sum_i u_{ij}},\quad
        \gamma = \frac{n d}{2\sum_{i,j} u_{ij}\lVert x_i - v_j\rVert^2} .

    This is the EM algorithm of an isotropic Gaussian mixture with equal
    weights and a shared variance :math:`1/(2\gamma)`, which is how the
    reference describes it ("a simplified generalized Gaussian mixture
    model"). Each update decreases the objective, so
    ``objective_trajectory_`` is non-increasing.

    Parameters
    ----------
    n_clusters : int, default=3
    max_iter : int, default=100
    tol : float, default=1e-5
        Stop when the objective changes by at most ``tol`` relative to its
        magnitude.
    random_state : int, numpy Generator or None, default=None
        Seed for the k-means++ initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    centers_ : ndarray of shape (n_clusters, n_features)
    gamma_ : float
        The learned precision :math:`\gamma`.
    objective_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    Q. Chen, W. Yu, F. Nie and X. Li. Adaptive fuzzy c-means with graph
    embedding. arXiv:2405.13427, 2024.

    Notes
    -----
    Earlier releases implemented plain fuzzy c-means with a fuzzifier ``m``
    under this name; ``m`` has no role in the reference model and has been
    removed.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X) -> AFCMSimple:
        """Fit on a feature matrix of shape ``(n_samples, n_features)``."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        K = check_n_clusters(self.n_clusters, X.shape[0], estimator=name)
        V = _init_centers_kpp(X, K, check_generator(self.random_state))
        labels = np.argmin(_euclidean_dist2(X, V), axis=1)
        U = np.zeros((X.shape[0], K))
        U[np.arange(X.shape[0]), labels] = 1.0

        trajectory = []
        J_prev = np.inf
        for _ in range(self.max_iter):
            U, V, gamma, J = adaptive_fcm_step(X, U)
            trajectory.append(J)
            if objective_converged(J_prev, J, self.tol):
                break
            J_prev = J

        # Final prototypes and precision consistent with the returned U.
        weights = np.maximum(U.sum(axis=0), np.finfo(float).tiny)
        self.centers_ = (U.T @ X) / weights[:, None]
        self.gamma_ = gamma
        self.memberships_ = self._predict_from(X)
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = len(trajectory)
        return self

    def _predict_from(self, X):
        scores = -self.gamma_ * _euclidean_dist2(X, self.centers_)
        scores -= scores.max(axis=1, keepdims=True)
        U = np.exp(scores)
        return U / U.sum(axis=1, keepdims=True)

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return self._predict_from(X)
