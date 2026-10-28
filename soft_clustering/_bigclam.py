# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Cluster affiliation model for big networks (Yang and Leskovec, 2013)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_generator, check_n_clusters, check_scalar

_X_FLOOR = 1e-10  # keeps log(1 - exp(-x)) finite on edges with x -> 0


def _adjacency(adj, name):
    A = sp.csr_matrix(adj, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"{name}: adjacency matrix must be square; got shape {A.shape}."
        )
    if not np.isfinite(A.data).all():
        raise ValueError(f"{name}: adjacency matrix contains NaN or infinite values.")
    A = A.maximum(A.T)  # undirected
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return A.tocoo()


@typechecked
class BIGCLAM(BaseSoftClusterer):
    r"""Cluster affiliation model for big networks (BigCLAM).

    Every node :math:`u` has a non-negative affiliation vector :math:`F_u \in
    \mathbb{R}_{\ge 0}^K`, and an edge appears between :math:`u` and :math:`v`
    with probability :math:`1 - \exp(-F_u^\top F_v)`, so nodes sharing
    communities are more likely to connect. BigCLAM maximises the
    log-likelihood

    .. math::

        \ell(F) = \sum_{(u,v) \in E} \log\big(1 - e^{-F_u^\top F_v}\big)
                - \sum_{(u,v) \notin E} F_u^\top F_v

    by projected gradient ascent, using

    .. math::

        \nabla_{F_u}\ell = \sum_{v \in N(u)} F_v
            \frac{e^{-F_u^\top F_v}}{1 - e^{-F_u^\top F_v}}
          - \sum_{v \notin N(u),\, v \ne u} F_v .

    As in the reference, nodes are updated one at a time (block coordinate
    ascent), each with its own backtracking line search on the part of
    :math:`\ell` that depends on :math:`F_u`; the non-neighbour sum is
    obtained from a running total :math:`\sum_v F_v`, so a sweep costs
    :math:`O(|E| K)`. Every accepted step increases :math:`\ell`, so
    ``log_likelihood_trajectory_`` is non-decreasing.

    ``memberships_`` holds the affiliation strengths :math:`F`, which are not
    normalised: a node may belong strongly to several communities or to none.

    **Deviation from the reference.** Yang and Leskovec initialise
    :math:`F` from locally minimal neighbourhoods (conductance); SCPP
    initialises it uniformly at random in :math:`[0, 1)`, which may reach a
    different local optimum. Nodes are visited in a random order each sweep.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of communities (``n_communities`` is accepted as an alias).
    n_nodes : int or None, default=None
        Ignored; the node count is read from the adjacency matrix. Kept for
        backward compatibility.
    max_iter : int, default=200
    learning_rate : float, default=1.0
        Initial step size of every node's line search.
    tol : float, default=1e-5
        Stop when the relative log-likelihood gain is below ``tol``.
    random_state : int, numpy Generator or None, default=None

    Attributes
    ----------
    memberships_ : ndarray of shape (n_nodes, n_clusters)
        Affiliation matrix :math:`F` (also ``F``).
    log_likelihood_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    J. Yang and J. Leskovec. Overlapping community detection at scale: a
    nonnegative matrix factorization approach. In *Proc. 6th ACM
    International Conference on Web Search and Data Mining (WSDM)*,
    pp. 587--596, 2013. doi:10.1145/2433396.2433471

    Notes
    -----
    Earlier releases subtracted the sum over *all* other nodes in the
    gradient, omitting the neighbour correction, and drew the initial
    affiliations from NumPy's global random state inside the constructor.
    """

    _partition_constrained = False
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    # Fitted attributes, set by ``fit`` (not before).
    F: np.ndarray

    def __init__(
        self,
        n_nodes: int | None = None,
        n_clusters: int = 3,
        max_iter: int = 200,
        learning_rate: float = 1.0,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.n_communities = n_clusters
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _loglik(F, rows, cols):
        x = np.maximum(np.einsum("ij,ij->i", F[rows], F[cols]), _X_FLOOR)
        total = F.sum(axis=0)
        all_pairs = float(total @ total - np.einsum("ij,ij->", F, F))  # u != v
        edge_x = float(x.sum())
        return float(np.log(-np.expm1(-x)).sum()) - (all_pairs - edge_x)

    @staticmethod
    def _gradient(F, rows, cols, n):
        x = np.maximum(np.einsum("ij,ij->i", F[rows], F[cols]), _X_FLOOR)
        # exp(-x)/(1-exp(-x)) + 1 = 1/(1-exp(-x)): the neighbour term plus
        # the correction that removes neighbours from the non-edge sum.
        w = 1.0 / (-np.expm1(-x))
        S = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
        return S @ F - (F.sum(axis=0)[None, :] - F)

    def fit(self, adj) -> BIGCLAM:
        """Fit on a (dense or sparse) ``(n_nodes, n_nodes)`` adjacency matrix.

        The graph is treated as undirected and unweighted; self-loops are
        ignored.
        """
        name = type(self).__name__
        A = _adjacency(adj, name)
        n = A.shape[0]
        K = check_n_clusters(self.n_clusters, n, estimator=name)
        step = check_scalar(
            self.learning_rate,
            "learning_rate",
            estimator=name,
            lower=0.0,
            lower_inclusive=False,
        )
        rows, cols = A.row, A.col  # both directions of every undirected edge

        rng = check_generator(self.random_state)
        F = rng.random((n, K))
        A = A.tocsr()
        indptr, indices = A.indptr, A.indices
        total = F.sum(axis=0)

        def node_objective(f, neigh_F, others):
            x = np.maximum(neigh_F @ f, _X_FLOOR)
            return float(np.log(-np.expm1(-x)).sum() - f @ others)

        ll = self._loglik(F, rows, cols)
        trace = [ll]
        for _ in range(self.max_iter):
            for u in rng.permutation(n):
                neigh = indices[indptr[u] : indptr[u + 1]]
                neigh_F = F[neigh]
                f = F[u]
                # sum of F_v over non-neighbours v != u
                others = total - f - neigh_F.sum(axis=0)
                x = np.maximum(neigh_F @ f, _X_FLOOR)
                grad = neigh_F.T @ (np.exp(-x) / -np.expm1(-x)) - others
                base = node_objective(f, neigh_F, others)
                t = step
                for _ in range(20):
                    f_new = np.maximum(f + t * grad, 0.0)
                    if node_objective(
                        f_new, neigh_F, others
                    ) >= base + 1e-4 * t * float(grad @ (f_new - f)) / max(t, 1e-300):
                        total += f_new - f
                        F[u] = f_new
                        break
                    t *= 0.5
            ll_new = self._loglik(F, rows, cols)
            trace.append(ll_new)
            if ll_new - ll <= self.tol * max(abs(ll_new), 1.0):
                ll = ll_new
                break
            ll = ll_new

        self.F = F
        self.memberships_ = F
        self.log_likelihood_trajectory_ = np.array(trace)
        self.n_iter_ = len(trace) - 1
        return self

    def get_membership(self) -> np.ndarray:
        """Return the affiliation matrix ``F``, shape ``(n_nodes, n_clusters)``."""
        self._check_fitted()
        return self.F
