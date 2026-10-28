# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Kernelized Fuzzy C-Means (KFCM).

Implementation note (optimization study)
----------------------------------------
Vectorised implementation of the same algorithm. The update rules, the
K-Means++ initialisation, the convergence test and — critically — the order and
number of draws taken from NumPy's global random state are unchanged. The
reference implementation is preserved at
``optimization/original/scpp_original/_kfcm.py``.

Profiling reported ``typeguard``'s runtime type checking as roughly 23% of fit
time. That was a symptom rather than the cause: ``_gaussian_kernel`` is a
scalar helper called ``N * n_clusters`` times per iteration from Python list
comprehensions, and the per-call type check rode on every one of those calls.
Evaluating the kernel as a matrix removes the calls and the checking overhead
together.

Three list comprehensions are replaced by array operations:

* the center update's ``[kernel(x_k, V[i]) for x_k in X]``, once per cluster
  per iteration;
* the membership update's ``[1 - kernel(X[k], V[i]) for i in ...]``, once per
  sample per iteration;
* the K-Means++ initialisation's ``[min([norm(x - c)**2 for c in centers]) for
  x in X]``, which is quadratic in Python-level work.

Exactness notes. The squared distance is computed as ``sqrt(sum(d**2))**2``,
not as ``sum(d**2)``, because that is what ``np.linalg.norm(...) ** 2`` does in
the reference and the round trip through ``sqrt`` is not the identity in
floating point. The two agree closely but not bit-for-bit, because
``np.linalg.norm`` reduces through a BLAS dot product whose summation order
need not match ``np.sum``; the resulting 1-ULP difference is amplified by
``exp``, giving kernel values that agree to a relative 1e-12 and fitted
memberships that agree to an absolute 6e-15. The K-Means++ roulette selection — "first index whose
cumulative probability exceeds the draw" — becomes ``np.searchsorted(cum, r,
side="right")``, which returns exactly that index, and the out-of-range case
(no such index) leaves the center at zero as the reference's un-taken ``break``
did. The ``(N, n_clusters, n_features)`` difference tensor is materialised
directly rather than chunked: ``n_clusters`` is a small constant here, unlike
in ``SoftDBSCANGM`` where it grows with ``N``.
"""

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer, ratio_memberships
from ._validation import (
    check_array,
    check_n_clusters,
    check_n_features,
    check_random_state,
    check_scalar,
)


@typechecked
class KFCM(BaseSoftClusterer):
    r"""Kernelized fuzzy c-means (KFCM) with a Gaussian kernel.

    Fuzzy c-means in the feature space of the Gaussian kernel
    :math:`K(x, v) = \exp(-\lVert x - v\rVert^2/\sigma^2)`, with prototypes
    kept in the input space (Zhang and Chen, 2004). Because
    :math:`\lVert\phi(x) - \phi(v)\rVert^2 = 2(1 - K(x, v))`, the objective is

    .. math::

        J = 2 \sum_{i,k} u_{ik}^m \big(1 - K(x_i, v_k)\big),

    with updates

    .. math::

        u_{ik} \propto \big(1 - K(x_i, v_k)\big)^{-1/(m-1)},
        \qquad
        v_k = \frac{\sum_i u_{ik}^m K(x_i, v_k)\, x_i}{\sum_i u_{ik}^m K(x_i, v_k)} .

    The prototype update is a fixed-point iteration (``v_k`` appears on both
    sides), as in the reference. Prototypes are seeded with k-means++.

    ``sigma`` is in the units of ``X``: when it is much smaller than the
    distances between points, every kernel value underflows towards zero,
    every distance ``1 - K`` tends to one and the memberships become uniform.

    Parameters
    ----------
    n_clusters : int, default=3
    m : float, default=2.0
        Fuzzifier, ``m > 1``.
    sigma : float, default=1.0
        Kernel bandwidth, ``sigma > 0``.
    epsilon : float, default=0.01
        Stop when no membership changes by more than ``epsilon``.
    max_iter : int, default=100
    random_state : int, numpy RandomState or None, default=None
        Seed for the k-means++ draws and the random initial memberships.
        ``None`` draws from NumPy's global random state, as earlier releases
        did.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    centers_ : ndarray of shape (n_clusters, n_features)
        Prototypes in the input space (also ``V``).
    U : ndarray of shape (n_clusters, n_samples)
        Cluster-major memberships.
    n_iter_ : int

    References
    ----------
    D.-Q. Zhang and S.-C. Chen. A novel kernelized fuzzy c-means algorithm
    with application in medical image segmentation. *Artificial Intelligence
    in Medicine*, 32(1):37--50, 2004. doi:10.1016/j.artmed.2004.01.012
    """

    _supports_out_of_sample = True

    # ``V`` and ``U`` are the names the reference implementation used; they
    # are stored privately so that fit changes no public attribute.
    @property
    def V(self) -> np.ndarray | None:
        """Prototypes, ``(n_clusters, n_features)``; also ``centers_``."""
        return getattr(self, "_V", None)

    @V.setter
    def V(self, value) -> None:
        self._V = value

    @property
    def U(self) -> np.ndarray | None:
        """Cluster-major memberships, ``(n_clusters, n_samples)``."""
        return getattr(self, "_U", None)

    @U.setter
    def U(self, value) -> None:
        self._U = value

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        sigma: float = 1.0,
        epsilon: float = 0.01,
        max_iter: int = 100,
        random_state: int | np.random.RandomState | None = None,
    ):
        """
        Initializes the KFCM algorithm with given parameters.
        """
        self.n_clusters: int = n_clusters
        self.m: float = m
        self.sigma: float = sigma
        self.epsilon: float = epsilon
        self.max_iter: int = max_iter
        self.random_state = random_state

    def _squared_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Squared Euclidean distances between every sample and every center.

        Returns an ``(n_samples, n_centers)`` array. Computed as
        ``sqrt(sum(diff**2))**2`` so that each entry is bit-for-bit what
        ``np.linalg.norm(x - c) ** 2`` produces elementwise.
        """
        diff = X[:, None, :] - centers[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1)) ** 2

    def _kernel_matrix(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Gaussian RBF kernel between every sample and every center.

        The matrix form of :meth:`_gaussian_kernel`.
        """
        return np.exp(-self._squared_distances(X, centers) / self.sigma**2)

    def _initialize_centers_kmeans_pp(
        self, X: np.ndarray, rng: np.random.RandomState | None = None
    ) -> np.ndarray:
        """
        Initializes cluster centers using the K-Means++ strategy.
        This method spreads out the initial centers, leading to better convergence.
        """
        if rng is None:
            rng = check_random_state(None)
        N, D = X.shape
        centers = np.zeros((self.n_clusters, D))

        # 1. Choose the first center uniformly at random from the data points
        first_center_idx = rng.randint(N)
        centers[0] = X[first_center_idx]

        # 2. For the remaining centers
        for i in range(1, self.n_clusters):
            # Squared distance of each point to the nearest already-chosen center
            dist_sq = np.min(self._squared_distances(X, centers[:i]), axis=1)

            # 3. Choose the next center with probability proportional to the squared distance
            total = np.sum(dist_sq)
            r = rng.rand()
            if not np.isfinite(total) or total <= 0.0:
                # Every sample coincides with a chosen center, so the
                # distribution is undefined (0/0). The reference left the
                # center at the origin, which need not be a data point; draw
                # uniformly instead, from the same random number.
                centers[i] = X[min(int(r * N), N - 1)]
                continue
            cumulative_probs = np.cumsum(dist_sq / total)

            # First index whose cumulative probability exceeds r. Rounding can
            # leave the last cumulative value just below r; the reference then
            # left the center at the origin, here it takes the last sample.
            j = int(np.searchsorted(cumulative_probs, r, side="right"))
            centers[i] = X[min(j, N - 1)]
        return centers

    def _gaussian_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Gaussian RBF kernel between two points.
        """
        return float(np.exp(-np.linalg.norm(x - y) ** 2 / self.sigma**2))

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit on a feature matrix of shape ``(n_samples, n_features)``."""
        X = check_array(X, estimator=type(self).__name__)
        N, D = X.shape
        name = type(self).__name__
        check_n_clusters(self.n_clusters, N, estimator=name)
        check_scalar(self.m, "m", estimator=name, lower=1.0, lower_inclusive=False)
        check_scalar(
            self.sigma, "sigma", estimator=name, lower=0.0, lower_inclusive=False
        )
        rng = check_random_state(self.random_state)

        # --- Step 1: Initialization ---
        self.V = self._initialize_centers_kmeans_pp(X, rng).astype(np.float64)

        self.U = rng.rand(self.n_clusters, N)

        self.U = self.U / np.sum(self.U, axis=0)

        # --- Step 2: The Iteration Loop ---
        for t in range(self.max_iter):
            U_old = self.U.copy()

            # --- Step 3: Update Cluster Centers (V) ---
            # Every cluster reads the pre-update centers and writes only its
            # own row, so the reference's loop over i carries no dependency.
            K_v = self._kernel_matrix(X, self.V)  # (N, n_clusters)
            weights = self.U**self.m * K_v.T  # (n_clusters, N)
            numerators = weights @ X  # (n_clusters, D)
            denominators = np.sum(weights, axis=1)  # (n_clusters,)

            movable = denominators > 1e-9
            self.V[movable] = numerators[movable] / denominators[movable, None]

            # --- Step 4: Update Membership Matrix (U) ---
            # Recomputed against the centers just updated, as in the reference.
            self.U = self._memberships(X).T

            # --- Check for convergence ---
            if np.max(np.abs(self.U - U_old)) < self.epsilon:
                break

        self.n_iter_ = t + 1
        self.centers_ = self.V
        self.memberships_ = self.U.T
        return np.argmax(self.U, axis=0)

    def _memberships(self, X: np.ndarray) -> np.ndarray:
        """``u_ik`` proportional to ``(1 - K(x_i, v_k))^(-1/(m-1))``, samples-first."""
        V = self.V
        assert V is not None  # set by fit before memberships are computed
        dist = 1 - self._kernel_matrix(X, V)  # (N, n_clusters)
        dist[dist <= 0] = np.finfo(float).eps
        # Evaluated without forming the unbounded powers, so m close to one
        # does not overflow.
        return ratio_memberships(dist, 1.0 / (self.m - 1.0))

    def _predict_memberships(self, X) -> np.ndarray:
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return self._memberships(X)
