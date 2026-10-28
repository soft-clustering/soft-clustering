# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Kernel-based fuzzy competitive learning clustering (K-FCCL)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_n_features,
    check_scalar,
)

#: Above this many samples the kernel matrix is not stored (see ``fit``).
_CACHE_KERNEL_UP_TO = 5000


def _gibbs(D: np.ndarray, lam: float) -> np.ndarray:
    """Row-wise ``softmax(-lam * D)``."""
    scores = -lam * D
    scores -= scores.max(axis=-1, keepdims=True)
    np.exp(scores, out=scores)
    return scores / scores.sum(axis=-1, keepdims=True)


@typechecked
class KFCCL(BaseSoftClusterer):
    r"""Kernel-based fuzzy competitive learning clustering (K-FCCL).

    Competitive learning in the feature space of the Gaussian kernel
    :math:`K(x, y) = \exp(-\gamma\lVert x - y\rVert^2)`. Every prototype is a
    kernel expansion :math:`v_i = \sum_j g_{ij}\, \phi(x_j)`, so its distance
    to a mapped sample is available through the kernel trick,

    .. math::

        D_{ik} = \lVert \phi(x_k) - v_i\rVert^2
               = K_{kk} - 2 \sum_j g_{ij} K_{jk} + \sum_{j,l} g_{ij} g_{il} K_{jl}.

    Samples are presented one at a time in a random order. The presented
    sample receives fuzzy (Gibbs) memberships
    :math:`u_{ik} \propto \exp(-\lambda D_{ik})`, and every prototype moves
    towards it in proportion to its membership,

    .. math::

        v_i \leftarrow v_i + a_t\, u_{ik}\, \big(\phi(x_k) - v_i\big),
        \qquad\text{i.e.}\qquad
        g_i \leftarrow (1 - a_t u_{ik})\, g_i + a_t u_{ik}\, e_k ,

    with a learning rate :math:`a_t = a_0 / (1 + t)` decaying over epochs
    :math:`t`. Prototypes start at distinct samples chosen by kernel
    k-means++ seeding, which breaks the symmetry between clusters. The fit
    stops when no sample's membership changes by more than ``epsilon``
    between epochs. ``memberships_`` are the Gibbs memberships of all samples
    under the final prototypes.

    **Verification status.** This is the standard online formulation of
    kernel fuzzy competitive learning; the exact update and schedule of
    Mizutani and Miyamoto (2005) could not be checked against the paper,
    and the implementation has not been compared with an independent one.

    **Cost.** An epoch takes :math:`O(K n^2)` time. Up to 5000 samples the
    kernel matrix is stored (at most 200 MB); above that, kernel rows are
    computed on demand, so memory stays :math:`O(K n)` at the price of
    :math:`O(n^2 d)` extra work per epoch. Both give the same result.

    Parameters
    ----------
    n_clusters : int, default=2
    lambda_ : float, default=10.0
        Inverse temperature :math:`\lambda > 0` of the memberships, in units
        of the reciprocal of a feature-space squared distance (which lies in
        ``[0, 2]`` for the Gaussian kernel). Larger is crisper.
    gamma : float, default=1.0
        Kernel parameter :math:`\gamma > 0`, in units of ``1/||X||^2``.
    epsilon : float, default=1e-4
        Convergence threshold on the membership change between epochs.
    max_iter : int, default=50
        Maximum number of epochs. The learning rate has decayed to 1% of its
        initial value by epoch 100, so later epochs change little.
    learning_rate : float, default=0.5
        Initial learning rate :math:`a_0 \in (0, 1]`.
    random_state : int, numpy Generator or None, default=None
        Seed for the seeding and for the presentation order.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    coefficients_ : ndarray of shape (n_clusters, n_samples)
        Expansion coefficients :math:`g_{ij}` of the prototypes.
    U : ndarray of shape (n_clusters, n_samples)
        Cluster-major memberships, kept for backward compatibility.
    n_iter_ : int

    References
    ----------
    K. Mizutani and S. Miyamoto. Kernel-based fuzzy competitive learning
    clustering. In *Proc. 14th IEEE International Conference on Fuzzy
    Systems (FUZZ-IEEE '05)*, pp. 636--639, 2005.
    doi:10.1109/FUZZY.2005.1452468

    Notes
    -----
    Earlier releases started every prototype from the same near-uniform
    state and applied an essentially identical batch update to all of them,
    so the clusters never separated: on three well-separated Gaussian blobs
    every membership was 1/3 and the adjusted Rand index was 0. That
    implementation is preserved for the optimisation study in
    ``optimization/original/scpp_original/_kfccl.py``.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ()
    _supports_out_of_sample = True

    @property
    def U(self) -> np.ndarray | None:
        """Cluster-major memberships, ``(n_clusters, n_samples)``."""
        U = getattr(self, "memberships_", None)
        return None if U is None else U.T

    def __init__(
        self,
        n_clusters: int = 2,
        lambda_: float = 10.0,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
        max_iter: int = 50,
        learning_rate: float = 0.5,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.lambda_ = lambda_
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    @staticmethod
    def _seed(kernel_rows, n, n_clusters, rng):
        """Kernel k-means++: distinct samples, far apart in feature space.

        ``kernel_rows(idx)`` returns the kernel rows of the samples ``idx``;
        the Gaussian kernel has a unit diagonal.
        """
        chosen = [int(rng.integers(n))]
        for _ in range(1, n_clusters):
            d = 2.0 - 2.0 * kernel_rows(np.array(chosen)).T
            d_min = np.maximum(d.min(axis=1), 0.0)
            d_min[chosen] = 0.0
            total = d_min.sum()
            if total <= 0:
                remaining = np.setdiff1d(np.arange(n), chosen)
                chosen.append(int(rng.choice(remaining)))
            else:
                chosen.append(int(rng.choice(n, p=d_min / total)))
        return chosen

    def fit(self, X) -> KFCCL:
        """Fit on a feature matrix of shape ``(n_samples, n_features)``."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        n = X.shape[0]
        c = check_n_clusters(self.n_clusters, n, estimator=name)
        lam = check_scalar(
            self.lambda_, "lambda_", estimator=name, lower=0.0, lower_inclusive=False
        )
        gamma = check_scalar(
            self.gamma, "gamma", estimator=name, lower=0.0, lower_inclusive=False
        )
        a0 = check_scalar(
            self.learning_rate,
            "learning_rate",
            estimator=name,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        rng = check_generator(self.random_state)

        # Kernel rows are computed when needed instead of storing the
        # (n, n) kernel matrix, so memory is O(n * n_clusters).
        sq = np.einsum("ij,ij->i", X, X)

        def kernel_rows(idx):
            d2 = sq[idx, None] + sq[None, :] - 2.0 * (X[idx] @ X.T)
            return np.exp(-gamma * np.maximum(d2, 0.0))

        if n <= _CACHE_KERNEL_UP_TO:
            # Small problems: one (n, n) kernel (at most 200 MB) is faster
            # than recomputing a row for every presented sample.
            K = kernel_rows(np.arange(n))

            def kernel_rows(idx):  # noqa: F811 - cached replacement
                return K[idx]

        diag = np.ones(n)  # K(x, x) = 1 for the Gaussian kernel
        seeds = self._seed(kernel_rows, n, c, rng)
        G = np.zeros((c, n))
        G[np.arange(c), seeds] = 1.0
        GK = kernel_rows(np.array(seeds))  # (c, n): sum_j g_ij K_jk
        q = np.einsum("ij,ij->i", GK, G)  # ||v_i||^2

        # G and GK = G K are stored as scale[:, None] * (G_t, GK_t). Shrinking a
        # prototype then only rescales ``scale`` (O(c)) instead of both
        # (c, n) matrices; adding the presented sample touches one column of
        # G_t and adds one scaled kernel row to GK_t.
        G_t, GK_t, scale = G, GK, np.ones(c)

        U_prev = _gibbs((diag[None, :] - 2 * GK + q[:, None]).T, lam)
        for epoch in range(self.max_iter):
            a = a0 / (1.0 + epoch)
            for k in rng.permutation(n):
                gk = scale * GK_t[:, k]  # <v_i, phi(x_k)>
                d = diag[k] - 2 * gk + q
                u = np.exp(-lam * (d - d.min()))
                step = a * u / u.sum()
                shrink = 1.0 - step
                # q_i' = s^2 q_i + 2 s b <v_i, phi(x_k)> + b^2 K_kk
                q = shrink**2 * q + 2 * shrink * step * gk + step**2 * diag[k]
                scale *= shrink
                add = step / scale
                G_t[:, k] += add
                GK_t += add[:, None] * kernel_rows(np.array([k]))
                if scale.min() < 1e-150:  # fold the scale back before underflow
                    G_t *= scale[:, None]
                    GK_t *= scale[:, None]
                    scale = np.ones(c)
            GK = scale[:, None] * GK_t
            U = _gibbs((diag[None, :] - 2 * GK + q[:, None]).T, lam)  # (n, c)
            converged = np.max(np.abs(U - U_prev)) < self.epsilon
            U_prev = U
            if converged:
                break
        G = scale[:, None] * G_t

        self._X_fit = X
        self.coefficients_ = G
        self._q = q
        self.memberships_ = U
        self.n_iter_ = epoch + 1
        return self

    def _predict_memberships(self, X) -> np.ndarray:
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self._X_fit.shape[1], estimator=type(self).__name__)
        Kx = rbf_kernel(self._X_fit, X, gamma=self.gamma)  # (n_fit, n_new)
        D = 1.0 - 2 * (self.coefficients_ @ Kx) + self._q[:, None]
        return _gibbs(D.T, self.lambda_)
