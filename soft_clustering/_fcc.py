# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Fuzzy color clustering (Kim and Lee, 2024)."""

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
)


def _fcc_memberships(X: np.ndarray, centers: np.ndarray, jnd: float) -> np.ndarray:
    """Membership function of the fuzzy color model (Kim and Lee, Eq. 7).

    A color within ``jnd`` of a center belongs to that fuzzy color with degree
    one and to every other color with degree zero (shared equally if it lies
    inside several balls). Otherwise the degree is the inverse-ratio rule
    ``(sum_j delta_i / delta_j)^-1`` on the surface distances
    ``delta_i = ||x - c_i|| - jnd``.
    """
    rho = np.sqrt(_euclidean_dist2(X, centers))
    delta = rho - jnd
    inside = delta <= 0
    U = np.empty_like(rho)

    core = inside.any(axis=1)
    if core.any():
        hits = inside[core].astype(np.float64)
        U[core] = hits / hits.sum(axis=1, keepdims=True)
    rest = ~core
    if rest.any():
        inv = 1.0 / delta[rest]  # delta > 0 here
        U[rest] = inv / inv.sum(axis=1, keepdims=True)
    return U


@typechecked
class FCC(BaseSoftClusterer):
    r"""Fuzzy color clustering (FCC).

    Represents each cluster as a *fuzzy color*: a ball of radius ``jnd`` (the
    just-noticeable difference) around a center in CIELAB space. With the
    surface distance :math:`\delta(\tilde c_i, x) = \lVert x - c_i\rVert -
    \mathrm{jnd}`, the membership of a color :math:`x` is

    .. math::

        \mu_i(x) = \begin{cases}
            1 & \delta(\tilde c_i, x) \le 0,\\
            0 & \delta(\tilde c_j, x) \le 0 \text{ for some } j \ne i,\\
            \Big[\sum_j \delta(\tilde c_i, x) / \delta(\tilde c_j, x)\Big]^{-1}
              & \text{otherwise,}
        \end{cases}

    and centers are updated as :math:`c_i = \sum_x \mu_i(x)\, x / \sum_x
    \mu_i(x)` (Kim and Lee, 2024, Eqs. 7, 17, 18). Rows of ``memberships_``
    sum to one.

    **Deviation from the reference.** The reference initialises the centers
    from the best-matching of thirteen predefined reference colors (Munsell
    hues and the lightness axis). That is specific to color data; SCPP uses
    k-means++ seeding so that the estimator applies to any three-dimensional
    (or other) feature space. The data are expected in CIELAB units when
    ``jnd`` is interpreted as a perceptual threshold.

    **Known behaviour.** Outside the balls the memberships decay only as the
    inverse distance, and centers are weighted by :math:`\mu` rather than a
    power of it, so the memberships are much less crisp than fuzzy c-means'
    and centers that no ball separates are drawn towards each other. On data
    whose clusters are not separated by ``jnd`` the partition can use fewer
    than ``n_clusters`` clusters; if ``jnd`` is large relative to the data,
    a single ball can absorb every point.

    Parameters
    ----------
    n_clusters : int, default=3
    jnd : float, default=20.0
        Radius of every fuzzy color, ``>= 0``, in the units of ``X``. The
        default is a perceptual threshold for CIELAB colors; for other data
        choose it relative to the spread of the data. With ``jnd = 0`` the
        memberships are inverse ratios of plain (unsquared) distances.
    max_iter : int, default=100
    tol : float, default=1e-5
        Stop when the centers move by less than ``tol`` (Frobenius norm).
    random_state : int, numpy Generator or None, default=None

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    centers_ : ndarray of shape (n_clusters, n_features)
    n_iter_ : int

    References
    ----------
    D.-W. Kim and K. H. Lee. Fuzzy color model and clustering algorithm for
    color clustering problem. arXiv:2407.06782, 2024.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        jnd: float = 20.0,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.jnd = jnd
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X) -> FCC:
        """Fit on colors of shape ``(n_samples, 3)`` (or any feature matrix)."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        K = check_n_clusters(self.n_clusters, X.shape[0], estimator=name)
        jnd = check_scalar(self.jnd, "jnd", estimator=name, lower=0.0)
        centers = _init_centers_kpp(X, K, check_generator(self.random_state))
        for it in range(self.max_iter):
            U = _fcc_memberships(X, centers, jnd)
            weights = U.sum(axis=0)
            new_centers = centers.copy()
            alive = weights > 0
            new_centers[alive] = (U[:, alive].T @ X) / weights[alive, None]
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break
        self.centers_ = centers
        self.memberships_ = _fcc_memberships(X, centers, jnd)
        self.n_iter_ = it + 1
        return self

    def fit_predict(self, X) -> np.ndarray:
        """Fit and return ``memberships_``."""
        return self.fit(X).memberships_

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return _fcc_memberships(X, self.centers_, self.jnd)
