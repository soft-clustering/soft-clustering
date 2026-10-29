# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Entropy-regularised fuzzy c-means (maximum-entropy clustering)."""

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
    check_scalar,
    objective_converged,
)


def _entropy_memberships(d2: np.ndarray, lam: float) -> np.ndarray:
    """``u_ik = exp(-d_ik^2/lam) / sum_j exp(-d_ij^2/lam)``, as a stable softmax."""
    # Shift before scaling so at least one score stays zero even for tiny lam.
    with np.errstate(over="ignore"):
        scores = -(d2 - d2.min(axis=1, keepdims=True)) / lam
    np.exp(scores, out=scores)
    return scores / scores.sum(axis=1, keepdims=True)


@typechecked
class ENTROPYFCM(BaseSoftClusterer):
    r"""Entropy-regularised fuzzy c-means (maximum-entropy c-means).

    Replaces the fuzzifier of fuzzy c-means with an entropy regulariser:

    .. math::

        J(U, V) = \sum_{i,k} u_{ik} \lVert x_i - v_k\rVert^2
                + \lambda \sum_{i,k} u_{ik} \log u_{ik},
        \qquad \sum_k u_{ik} = 1 .

    The stationary conditions give a Gibbs (softmax) membership and a
    membership-weighted mean,

    .. math::

        u_{ik} = \frac{\exp(-\lVert x_i - v_k\rVert^2/\lambda)}
                      {\sum_j \exp(-\lVert x_i - v_j\rVert^2/\lambda)},
        \qquad
        v_k = \frac{\sum_i u_{ik} x_i}{\sum_i u_{ik}} .

    :math:`\lambda` plays the role of a temperature: as :math:`\lambda \to 0`
    the method becomes k-means, and as :math:`\lambda \to \infty` every
    membership tends to :math:`1/K`. Unlike the FCM fuzzifier, it is in the
    units of squared distance, so it should be chosen relative to the
    within-cluster variance of the data. Each update minimises :math:`J` in
    one block of variables, so ``objective_trajectory_`` is non-increasing.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    entropy_weight : float, default=1.0
        Regularisation weight :math:`\lambda > 0`.
    max_iter : int, default=100
        Maximum number of sweeps.
    tol : float, default=1e-5
        Stop when the objective changes by at most ``tol`` relative to its
        magnitude.
    random_state : int, numpy Generator or None, default=None
        Seed for the k-means++ initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    centers_ : ndarray of shape (n_clusters, n_features)
    objective_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    R.-P. Li and M. Mukaidono. A maximum-entropy approach to fuzzy
    clustering. In *Proc. 4th IEEE International Conference on Fuzzy Systems
    (FUZZ-IEEE/IFES '95)*, pp. 2227--2232, 1995.

    Notes
    -----
    The paper accompanying SCPP cites Gupta, Datta and Das (2021) for this
    estimator, an evolutionary multi-objective method that is not implemented
    here. Earlier releases accepted ``entropy_weight`` but never used it and
    computed plain fuzzy c-means; the parameter ``m`` they also accepted has
    no role in this formulation and has been removed.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        entropy_weight: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.entropy_weight = entropy_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X) -> ENTROPYFCM:
        """Fit the model on a feature matrix of shape ``(n_samples, n_features)``."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        K = check_n_clusters(self.n_clusters, X.shape[0], estimator=name)
        lam = check_scalar(
            self.entropy_weight,
            "entropy_weight",
            estimator=name,
            lower=0.0,
            lower_inclusive=False,
        )

        V = _init_centers_kpp(X, K, check_generator(self.random_state))
        U = _entropy_memberships(_euclidean_dist2(X, V), lam)
        trajectory = []
        J_prev = np.inf
        for _ in range(self.max_iter):
            weights = U.sum(axis=0)
            V = (U.T @ X) / np.maximum(weights, np.finfo(float).tiny)[:, None]
            d2 = _euclidean_dist2(X, V)
            U = _entropy_memberships(d2, lam)
            entropy = float(np.sum(U * np.log(np.maximum(U, np.finfo(float).tiny))))
            J = float(np.sum(U * d2)) + lam * entropy
            trajectory.append(J)
            if objective_converged(J_prev, J, self.tol):
                break
            J_prev = J

        self.centers_ = V
        self.memberships_ = U
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = len(trajectory)
        return self

    def fit_predict(self, X) -> np.ndarray:
        """Fit and return ``memberships_``.

        Earlier releases returned ``(labels, U)``; the protocol-wide return
        value of ``fit_predict`` is the membership matrix.
        """
        return self.fit(X).memberships_

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return _entropy_memberships(
            _euclidean_dist2(X, self.centers_), self.entropy_weight
        )
