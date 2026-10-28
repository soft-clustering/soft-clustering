# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Centroid auto-fused hierarchical fuzzy c-means (Lin and Chen, 2021)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse.csgraph import connected_components
from typeguard import typechecked

from ._base import BaseSoftClusterer, ratio_memberships
from ._fcm import _euclidean_dist2
from ._validation import check_array, check_generator, check_scalar


def _pairwise(V):
    d = np.sqrt(np.maximum(_euclidean_dist2(V, V), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


#: Grid of dimensionless fusion strengths traced by ``alpha="auto"``.
_ALPHA_PATH = np.geomspace(1e-4, 10.0, 51)


def _stable_choice(counts, K0):
    """Index range of the cluster count that is stable over the widest range.

    ``counts[t]`` is the number of clusters at the ``t``-th (log-spaced) alpha.
    The starting count ``K0`` and the trivial single cluster are excluded; on
    a tie the larger count (less fusion) wins. Returns ``(first, last)``
    indices of that count's run, or ``None`` if no count qualifies.
    """
    runs = {}
    for t, k in enumerate(counts):
        if 1 < k < K0:
            first, _ = runs.get(k, (t, t))
            runs[k] = (first, t)
    if not runs:
        return None
    best = max(runs, key=lambda k: (runs[k][1] - runs[k][0], k))
    return runs[best]


def _solve(X, V, gamma, fuse_dist, max_iter, tol):
    """Majorisation-minimisation from centroids ``V`` at penalty ``gamma``."""
    trajectory = []
    history = []
    J_prev = np.inf
    for it in range(max_iter):
        U = ratio_memberships(_euclidean_dist2(X, V) + 1e-300, 1.0)
        W2 = U**2
        s = W2.sum(axis=0)
        B = W2.T @ X

        # One majorisation-minimisation step for the centroids.
        D = _pairwise(V)
        weights = np.where(D > 0, 1.0 / np.maximum(D, fuse_dist * 1e-3), 0.0)
        np.fill_diagonal(weights, 0.0)
        L = np.diag(weights.sum(axis=1)) - weights
        A = np.diag(np.maximum(s, 1e-300)) + gamma * L
        V = np.linalg.solve(A, B)

        # Fuse centroids that have met.
        D = _pairwise(V)
        close = (D < fuse_dist) & ~np.eye(len(V), dtype=bool)
        if close.any():
            n_comp, comp = connected_components(close, directed=False)
            mass = np.maximum(s, 1e-300)
            V = np.vstack(
                [
                    (mass[comp == c, None] * V[comp == c]).sum(axis=0)
                    / mass[comp == c].sum()
                    for c in range(n_comp)
                ]
            )
            history.append((it, len(V)))

        U = ratio_memberships(_euclidean_dist2(X, V) + 1e-300, 1.0)
        J = float(
            0.5 * np.sum(U**2 * _euclidean_dist2(X, V))
            + gamma * np.triu(_pairwise(V), 1).sum()
        )
        trajectory.append(J)
        if np.isfinite(J_prev) and abs(J_prev - J) <= tol * max(abs(J), 1e-300):
            break
        J_prev = J
    return V, history, trajectory


@typechecked
class CAFHFCM(BaseSoftClusterer):
    r"""Centroid auto-fused hierarchical fuzzy c-means (CAF-HFCM).

    Starts from many more centroids than clusters and lets a group-lasso
    penalty on centroid differences fuse them (Lin and Chen, 2021):

    .. math::

        \min_{U, V}\; \frac12 \sum_{i,j} u_{ij}^2 \lVert x_i - v_j\rVert^2
            + \gamma \sum_{k < l} \lVert v_k - v_l\rVert_2,
        \qquad \sum_j u_{ij} = 1 .

    The penalty uses the unsquared Euclidean norm, so differences are driven
    exactly to zero and centroids merge; the number of clusters is therefore
    an *output*. The fuzzifier is fixed at 2, as in the reference, and the
    membership update is the fuzzy c-means rule.

    **Optimisation.** The reference solves the centroid step with ADMM. SCPP
    minimises the same objective by majorisation--minimisation: the norm is
    majorised at the current centroids by
    :math:`\lVert a\rVert \le \lVert a\rVert^2 / (2\lVert a_0\rVert) +
    \lVert a_0\rVert/2`, which turns the centroid step into the linear system
    :math:`(\mathrm{diag}(s) + \gamma L_w) V = B` with :math:`s_j = \sum_i
    u_{ij}^2`, :math:`B_j = \sum_i u_{ij}^2 x_i` and :math:`L_w` the Laplacian
    with weights :math:`1/\lVert v_k - v_l\rVert`. Both steps decrease the
    objective. Centroids closer than ``fuse_tol`` (relative to the data's
    scale) are merged, which is the finite-precision counterpart of the
    exact zeros produced by ADMM's proximal step.

    **Choosing the fusion strength.** As in the reference, the default
    ``alpha="auto"`` traces a path of increasing fusion strengths, each fit
    warm-started from the centroids of the previous one, so that centroids
    that have fused stay fused and the cluster count decreases along the
    path. It reports the cluster count that is stable over the widest range
    of (log-spaced) strengths, ignoring the initial count and the single
    cluster at the end of the path, and returns the solution in the middle
    of that range. The path is kept in ``path_``.

    **Known behaviour.** The stability rule finds the planted number of
    clusters when the clusters are well separated relative to their spread.
    When some clusters are closer to each other than to the rest, the last
    merges of the path can be the most stable and a smaller count is
    reported. Inspect ``path_`` rather than trusting a single count.

    Parameters
    ----------
    alpha : float or "auto", default="auto"
        ``"auto"`` traces the fusion path and selects the stable cluster count
        (see above). A float is a single dimensionless fusion strength. The penalty weight is
        :math:`\gamma = \alpha\, n\, \sigma_X / K_0`, with :math:`\sigma_X` the
        root-mean-square deviation of the data from its mean and :math:`K_0`
        the initial number of centroids, so that the same ``alpha`` behaves
        similarly across data sets of different size and scale. Larger values
        fuse more aggressively and give fewer clusters. The path traced by
        ``"auto"`` covers :math:`10^{-4} \le \alpha \le 10`.
    n_init_clusters : int or None, default=None
        Initial number of centroids :math:`K_0`; ``None`` uses
        :math:`\lfloor 2\sqrt{n}\rfloor`, inside the range :math:`a \in [1, 3]`
        the reference recommends.
    max_iter : int, default=300
    tol : float, default=1e-6
        Stop when the relative change of the objective is below ``tol``.
    fuse_tol : float, default=1e-3
        Centroids closer than ``fuse_tol * sigma_X`` are merged.
    random_state : int, numpy Generator or None, default=None
        Seed for choosing the initial centroids among the samples.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, K)
    centers_ : ndarray of shape (K, n_features)
    n_clusters_ : int
        Number of clusters left after fusion.
    alpha_ : float
        Fusion strength of the returned solution.
    gamma_ : float
        Penalty weight :math:`\gamma` of the returned solution.
    path_ : list of tuple
        ``(alpha, n_clusters)`` for every strength traced when
        ``alpha="auto"``; a single entry otherwise.
    merge_history_ : list of tuple
        ``(iteration, n_centroids)`` after every merge, counted over all fits
        up to the returned one.
    objective_trajectory_ : ndarray
        Objective values of the fit at ``alpha_``.

    References
    ----------
    Y. Lin and S. Chen. A centroid auto-fused hierarchical fuzzy c-means
    clustering. *IEEE Transactions on Fuzzy Systems*, 29(7):2006--2017, 2021.
    doi:10.1109/TFUZZ.2020.2991306
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _determines_k = True

    def __init__(
        self,
        alpha: float | Literal["auto"] = "auto",
        n_init_clusters: int | None = None,
        max_iter: int = 300,
        tol: float = 1e-6,
        fuse_tol: float = 1e-3,
        random_state: int | np.random.Generator | None = None,
    ):
        self.alpha = alpha
        self.n_init_clusters = n_init_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.fuse_tol = fuse_tol
        self.random_state = random_state

    def fit(self, X) -> CAFHFCM:
        """Fit on a feature matrix of shape ``(n_samples, n_features)``."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        n = X.shape[0]
        auto = isinstance(self.alpha, str)
        if auto and self.alpha != "auto":
            raise ValueError(
                f"{name}: alpha must be a non-negative float or 'auto'; "
                f"got {self.alpha!r}."
            )
        if not auto:
            check_scalar(self.alpha, "alpha", estimator=name, lower=0.0)
        check_scalar(self.fuse_tol, "fuse_tol", estimator=name, lower=0.0)
        K0 = self.n_init_clusters
        if K0 is None:
            K0 = int(np.floor(2.0 * np.sqrt(n)))
        if K0 < 1:
            raise ValueError(f"{name}: n_init_clusters must be >= 1; got {K0}.")
        K0 = min(K0, n)

        rng = check_generator(self.random_state)
        sigma = float(np.sqrt(np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))))
        sigma = sigma if sigma > 0 else 1.0
        fuse_dist = self.fuse_tol * sigma
        alphas = _ALPHA_PATH if auto else [float(self.alpha)]

        V = X[rng.choice(n, size=K0, replace=False)].copy()
        fits = []
        merges_so_far: list[tuple[int, int]] = []
        offset = 0
        for a in alphas:
            V, merges, traj = _solve(
                X, V, a * n * sigma / K0, fuse_dist, self.max_iter, self.tol
            )
            merges_so_far.extend((offset + it, k) for it, k in merges)
            offset += len(traj)
            fits.append((float(a), V, list(merges_so_far), traj, offset))
            if len(V) == 1:
                break

        chosen = len(fits) - 1
        if auto:
            run = _stable_choice([len(f[1]) for f in fits], K0)
            if run is not None:
                chosen = (run[0] + run[1]) // 2
        alpha, V, history, trajectory, n_iter = fits[chosen]
        gamma = alpha * n * sigma / K0
        self.alpha_ = alpha
        self.path_ = [(f[0], len(f[1])) for f in fits]

        self.centers_ = V
        self.memberships_ = ratio_memberships(_euclidean_dist2(X, V) + 1e-300, 1.0)
        self.gamma_ = gamma
        self.merge_history_ = history
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = n_iter
        return self

    def fit_predict(self, X) -> np.ndarray:
        """Fit and return ``memberships_``."""
        return self.fit(X).memberships_
