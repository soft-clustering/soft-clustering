# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Bayesian NMF for overlapping community detection (Psorakis et al., 2011)."""

from __future__ import annotations

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import (
    check_array,
    check_n_clusters,
    check_random_state,
    check_scalar,
    objective_converged,
)

_EPS = 1e-10


@typechecked
class BayesianNMF(BaseSoftClusterer):
    r"""Bayesian non-negative matrix factorisation for overlapping communities.

    Models the (weighted) adjacency matrix as Poisson counts,
    :math:`V \sim \mathrm{Poisson}(WH)`, with half-normal priors of precision
    :math:`\beta_k` on column :math:`k` of :math:`W` and row :math:`k` of
    :math:`H`, and :math:`\beta_k \sim \mathrm{Gamma}(a, b)`. The MAP updates
    of Psorakis et al. (2011, Eqs. 17--19) are

    .. math::

        H &\leftarrow \frac{H}{W^\top \mathbf{1} + B H}
              \odot W^\top \frac{V}{WH}, \qquad
        W \leftarrow \frac{W}{\mathbf{1} H^\top + W B}
              \odot \frac{V}{WH} H^\top, \\
        \beta_k &\leftarrow \frac{N + a - 1}
              {\tfrac12\big(\sum_i w_{ik}^2 + \sum_j h_{kj}^2\big) + b},

    with :math:`B = \mathrm{diag}(\beta)`. Automatic relevance determination
    shrinks unneeded communities towards zero, so ``n_clusters`` is an upper
    bound and the number of communities with non-negligible weight is
    reported as ``n_effective_``.

    ``memberships_`` is the row-normalised :math:`W`,
    :math:`\pi_{ik} = w_{ik} / \sum_k w_{ik}` (Eq. 20), so rows sum to one; a
    node with no edges, whose row of :math:`W` vanishes, receives the uniform
    membership. The unnormalised affiliations are in ``affiliations_``.

    Parameters
    ----------
    n_clusters : int, default=3
        Maximum number of communities :math:`K`. Psorakis et al. use
        :math:`K = N` and let relevance determination prune.
    max_iter : int, default=100
    a, b : float, default=1.0
        Shape and rate of the Gamma prior on the precisions.
    tol : float, default=1e-5
        Stop when the relative change of the negative log-posterior is below
        ``tol``.
    random_state : int, numpy RandomState or None, default=None
        Seed for the random initial factors. ``None`` draws from NumPy's
        global random state, as earlier releases did.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_nodes, n_clusters)
    affiliations_ : ndarray of shape (n_nodes, n_clusters)
        The factor :math:`W` (also ``W``); ``H`` and ``beta`` are kept too.
    n_effective_ : int
        Number of communities whose column of :math:`W` carries more than
        ``1e-3`` of the largest column's mass.
    objective_trajectory_ : ndarray
        Negative log-posterior after every iteration.
    n_iter_ : int

    References
    ----------
    I. Psorakis, S. Roberts, M. Ebden and B. Sheldon. Overlapping community
    detection using Bayesian non-negative matrix factorization. *Physical
    Review E*, 83(6):066114, 2011. doi:10.1103/PhysRevE.83.066114

    Notes
    -----
    Earlier releases used :math:`\beta_k = \tfrac12(\lVert w_k\rVert^2 +
    \lVert h_k\rVert^2) + b/(N + a - 1)`, which inverts the relevance
    determination (large factors received large precisions), and returned
    the unnormalised :math:`W` as memberships.
    """

    _partition_constrained = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    # Fitted attributes, set by ``fit`` (not before).
    W: np.ndarray
    H: np.ndarray
    beta: np.ndarray

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        a: float = 1.0,
        b: float = 1.0,
        tol: float = 1e-5,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.a = a
        self.b = b
        self.tol = tol
        self.random_state = random_state

    def _neg_log_posterior(self, V, W, H, beta, N):
        WH = W @ H + _EPS
        nll = float(np.sum(WH - V * np.log(WH)))
        norms = (W**2).sum(axis=0) + (H**2).sum(axis=1)
        prior = float(np.sum(0.5 * beta * norms - N * np.log(beta)))
        hyper = float(np.sum(self.b * beta - (self.a - 1.0) * np.log(beta)))
        return nll + prior + hyper

    def fit(self, V) -> BayesianNMF:
        """Fit on a non-negative ``(n_nodes, n_nodes)`` adjacency matrix."""
        name = type(self).__name__
        V = check_array(V, estimator=name, name="V")
        if V.shape[0] != V.shape[1]:
            raise ValueError(f"{name}: V must be square; got shape {V.shape}.")
        if np.any(V < 0):
            raise ValueError(f"{name}: V must be non-negative.")
        N = V.shape[0]
        K = check_n_clusters(self.n_clusters, N, estimator=name)
        a = check_scalar(self.a, "a", estimator=name, lower=0.0, lower_inclusive=False)
        b = check_scalar(self.b, "b", estimator=name, lower=0.0)
        if N + a - 1.0 <= 0:
            raise ValueError(f"{name}: N + a - 1 must be positive.")

        rng = check_random_state(self.random_state)
        W = rng.rand(N, K)
        H = rng.rand(K, N)
        beta = np.ones(K)

        trace = []
        prev = np.inf
        for _ in range(self.max_iter):
            ratio = V / (W @ H + _EPS)
            H *= (W.T @ ratio) / (W.sum(axis=0)[:, None] + beta[:, None] * H + _EPS)
            ratio = V / (W @ H + _EPS)
            W *= (ratio @ H.T) / (H.sum(axis=1)[None, :] + W * beta[None, :] + _EPS)
            beta = (N + a - 1.0) / (
                0.5 * ((W**2).sum(axis=0) + (H**2).sum(axis=1)) + b + _EPS
            )
            obj = self._neg_log_posterior(V, W, H, beta, N)
            trace.append(obj)
            if objective_converged(prev, obj, self.tol):
                break
            prev = obj

        mass = W.sum(axis=0)
        self.n_effective_ = int(np.sum(mass > 1e-3 * max(float(mass.max()), _EPS)))
        row = W.sum(axis=1, keepdims=True)
        self.memberships_ = np.where(row > _EPS, W / np.maximum(row, _EPS), 1.0 / K)
        self.W = W
        self.H = H
        self.beta = beta
        self.affiliations_ = W
        self.objective_trajectory_ = np.array(trace)
        self.n_iter_ = len(trace)
        return self

    def get_membership(self) -> np.ndarray:
        """Return the unnormalised affiliation matrix ``W``."""
        self._check_fitted()
        return self.W
