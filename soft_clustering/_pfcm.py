# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Possibilistic fuzzy c-means (Pal, Pal, Keller and Bezdek, 2005)."""

from __future__ import annotations

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._fcm import _euclidean_dist2, _update_U_from_centers, fcm_fit
from ._pcm import _estimate_etas, _update_T_from_centers
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_n_features,
    check_scalar,
    data_scale,
)


@typechecked
class PFCM(BaseSoftClusterer):
    r"""Possibilistic fuzzy c-means (PFCM).

    Maintains a fuzzy membership matrix :math:`U` (rows sum to one) and a
    possibilistic typicality matrix :math:`T` (rows unconstrained), and
    minimises

    .. math::

        J = \sum_{i,k} \big(a\, u_{ik}^{m} + b\, t_{ik}^{\eta}\big)
            \lVert x_i - v_k\rVert^2
          + \sum_k \gamma_k \sum_i (1 - t_{ik})^{\eta} .

    The updates are the fuzzy c-means membership rule for :math:`U`,

    .. math::

        t_{ik} = \Big[1 + \big(b \lVert x_i - v_k\rVert^2/\gamma_k\big)^{1/(\eta-1)}\Big]^{-1},
        \qquad
        v_k = \frac{\sum_i (a u_{ik}^m + b t_{ik}^\eta)\, x_i}
                   {\sum_i (a u_{ik}^m + b t_{ik}^\eta)} .

    As Pal et al. prescribe, the run is initialised from fuzzy c-means and
    :math:`\gamma_k = \sum_i u_{ik}^m d_{ik}^2 / \sum_i u_{ik}^m` is computed
    once from the terminal FCM partition and held fixed, so each block update
    decreases :math:`J`. With ``typicality_weight = 0`` the method reduces to
    fuzzy c-means.

    ``memberships_`` exposes :math:`U`; the typicalities are in
    ``typicalities_``.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    membership_fuzzifier : float, default=2.0
        Exponent :math:`m > 1` on the memberships.
    typicality_fuzzifier : float, default=2.0
        Exponent :math:`\eta > 1` on the typicalities.
    membership_weight : float, default=1.0
        Relative weight :math:`a \ge 0` of the membership term.
    typicality_weight : float, default=1.0
        Relative weight :math:`b \ge 0` of the typicality term. ``a + b``
        must be positive.
    max_iter : int, default=150
        Maximum number of sweeps.
    tol : float, default=1e-5
        Stop when the prototypes move by less than ``tol`` times the data
        spread (root mean per-feature standard deviation), so the test does
        not depend on the units of ``X``.
    random_state : int, numpy Generator or None, default=None
        Seed for the fuzzy c-means initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy memberships :math:`U` (rows sum to one).
    typicalities_ : ndarray of shape (n_samples, n_clusters)
        Typicalities :math:`T` in ``(0, 1]``.
    centers_ : ndarray of shape (n_clusters, n_features)
    gamma_ : ndarray of shape (n_clusters,)
        The fixed scale parameters :math:`\gamma_k` (also ``gamma``).
    objective_trajectory_ : ndarray
    n_iter_ : int
    membership_matrix, typicality_matrix : ndarray of shape (n_clusters, n_samples)
        Cluster-major views of :math:`U` and :math:`T`, kept for backward
        compatibility.

    References
    ----------
    N. R. Pal, K. Pal, J. M. Keller and J. C. Bezdek. A possibilistic fuzzy
    c-means clustering algorithm. *IEEE Transactions on Fuzzy Systems*,
    13(4):517--530, 2005. doi:10.1109/TFUZZ.2004.840099
    """

    _partition_constrained = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        membership_fuzzifier: float = 2.0,
        typicality_fuzzifier: float = 2.0,
        membership_weight: float = 1.0,
        typicality_weight: float = 1.0,
        max_iter: int = 150,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.membership_fuzzifier = membership_fuzzifier
        self.typicality_fuzzifier = typicality_fuzzifier
        self.membership_weight = membership_weight
        self.typicality_weight = typicality_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # Short names used in the equations above, kept as read-only aliases.
    @property
    def gamma(self) -> np.ndarray | None:
        """Alias of ``gamma_``, kept for backward compatibility."""
        return getattr(self, "gamma_", None)

    @property
    def m(self) -> float:
        return self.membership_fuzzifier

    @property
    def eta(self) -> float:
        return self.typicality_fuzzifier

    @property
    def a(self) -> float:
        return self.membership_weight

    @property
    def b(self) -> float:
        return self.typicality_weight

    @property
    def cluster_centroids(self) -> np.ndarray | None:
        return self.centers_

    @property
    def membership_matrix(self) -> np.ndarray | None:
        return None if self.memberships_ is None else self.memberships_.T

    @property
    def typicality_matrix(self) -> np.ndarray | None:
        T = getattr(self, "typicalities_", None)
        return None if T is None else T.T

    def _validate(self, n_samples):
        name = type(self).__name__
        K = check_n_clusters(self.n_clusters, n_samples, estimator=name)
        m = check_scalar(
            self.membership_fuzzifier,
            "membership_fuzzifier",
            estimator=name,
            lower=1.0,
            lower_inclusive=False,
        )
        eta = check_scalar(
            self.typicality_fuzzifier,
            "typicality_fuzzifier",
            estimator=name,
            lower=1.0,
            lower_inclusive=False,
        )
        a = check_scalar(
            self.membership_weight, "membership_weight", estimator=name, lower=0.0
        )
        b = check_scalar(
            self.typicality_weight, "typicality_weight", estimator=name, lower=0.0
        )
        if a + b <= 0:
            raise ValueError(
                f"{name}: membership_weight + typicality_weight must be positive."
            )
        return K, m, eta, a, b

    def _typicalities(self, X, centers, eta, b):
        # t = 1 / (1 + (b d^2 / gamma)^(1/(eta-1))): the PCM rule with scale
        # gamma / b. b = 0 makes every typicality 1, which then carries zero
        # weight in the prototype update.
        if b == 0:
            return np.ones((X.shape[0], centers.shape[0]))
        return _update_T_from_centers(X, centers, self.gamma_ / b, eta)

    def fit(self, X) -> PFCM:
        """Fit the model on a feature matrix of shape ``(n_samples, n_features)``."""
        X = check_array(X, estimator=type(self).__name__)
        K, m, eta, a, b = self._validate(X.shape[0])
        rng = check_generator(self.random_state)

        U, centers, _ = fcm_fit(X, K, m, rng, max_iter=self.max_iter, tol=self.tol)
        gamma = _estimate_etas(X, U, centers, m)
        if np.any(gamma == 0):
            fallback = float(np.mean(_euclidean_dist2(X, centers)))
            gamma = np.where(gamma == 0, fallback, gamma)
        self.gamma_ = gamma

        scale = data_scale(X)
        trajectory = []
        for _ in range(self.max_iter):
            U = _update_U_from_centers(X, centers, m)
            T = self._typicalities(X, centers, eta, b)
            weights = a * U**m + b * T**eta
            new_centers = (weights.T @ X) / np.maximum(weights.sum(axis=0), 1e-300)[
                :, None
            ]

            d2 = _euclidean_dist2(X, new_centers)
            J = float(np.sum(weights * d2) + np.sum(gamma[None, :] * (1.0 - T) ** eta))
            trajectory.append(J)

            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            if shift < self.tol * scale:
                break

        self.centers_ = centers
        self.memberships_ = _update_U_from_centers(X, centers, m)
        self.typicalities_ = self._typicalities(X, centers, eta, b)
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = len(trajectory)
        return self

    def _check_X(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return X

    def _predict_memberships(self, X):
        return _update_U_from_centers(self._check_X(X), self.centers_, self.m)

    def predict_memberships(self, X) -> np.ndarray:
        """Fuzzy memberships of new data, cluster-major ``(n_clusters, n_new)``."""
        self._check_fitted()
        return self._predict_memberships(X).T

    def predict_typicalities(self, X) -> np.ndarray:
        """Typicalities of new data, cluster-major ``(n_clusters, n_new)``."""
        self._check_fitted()
        return self._typicalities(self._check_X(X), self.centers_, self.eta, self.b).T
