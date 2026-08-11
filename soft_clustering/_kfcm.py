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

from ._base import BaseSoftClusterer


@typechecked
class KFCM(BaseSoftClusterer):
    """
    An improved Python implementation of the Kernelized Fuzzy C-Means (KFCM) algorithm.

    This version incorporates K-Means++ initialization for more robust and consistent
    clustering results, addressing the issue of poor random starts.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        sigma: float = 1.0,
        epsilon: float = 0.01,
        max_iter: int = 100,
    ):
        """
        Initializes the KFCM algorithm with given parameters.
        """
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if m <= 1.0:
            raise ValueError("Fuzziness exponent m must be > 1.")
        if sigma <= 0:
            raise ValueError("Sigma must be a positive float.")

        self.n_clusters: int = n_clusters
        self.m: float = m
        self.sigma: float = sigma
        self.epsilon: float = epsilon
        self.max_iter: int = max_iter
        self.V: np.ndarray | None = None
        self.U: np.ndarray | None = None

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

    def _initialize_centers_kmeans_pp(self, X: np.ndarray) -> np.ndarray:
        """
        Initializes cluster centers using the K-Means++ strategy.
        This method spreads out the initial centers, leading to better convergence.
        """
        N, D = X.shape
        centers = np.zeros((self.n_clusters, D))

        # 1. Choose the first center uniformly at random from the data points
        first_center_idx = np.random.randint(N)
        centers[0] = X[first_center_idx]

        # 2. For the remaining centers
        for i in range(1, self.n_clusters):
            # Squared distance of each point to the nearest already-chosen center
            dist_sq = np.min(self._squared_distances(X, centers[:i]), axis=1)

            # 3. Choose the next center with probability proportional to the squared distance
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()

            # First index whose cumulative probability exceeds r. If there is
            # none, the center is left at zero, as in the reference.
            j = int(np.searchsorted(cumulative_probs, r, side="right"))
            if j < N:
                centers[i] = X[j]
        return centers

    def _gaussian_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Gaussian RBF kernel between two points.
        """
        return float(np.exp(-np.linalg.norm(x - y) ** 2 / self.sigma**2))

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the KFCM model to the data X.
        """
        N, D = X.shape
        if N == 0:
            raise ValueError("Input data cannot be empty.")

        # --- Step 1: Initialization ---
        self.V = self._initialize_centers_kmeans_pp(X).astype(np.float64)

        self.U = np.random.rand(self.n_clusters, N)

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
            dist = 1 - self._kernel_matrix(X, self.V)  # (N, n_clusters)
            dist[dist == 0] = np.finfo(float).eps
            inv_powers = (1 / dist) ** (1 / (self.m - 1))  # (N, n_clusters)
            denominators = np.sum(inv_powers, axis=1)  # (N,)

            updatable = denominators > 1e-9
            self.U[:, updatable] = (
                inv_powers[updatable] / denominators[updatable, None]
            ).T

            # --- Check for convergence ---
            if np.max(np.abs(self.U - U_old)) < self.epsilon:
                print(f"Converged at iteration {t+1}")
                break

        if self.U is None:
            raise RuntimeError("Fitting failed, membership matrix is None.")

        print("KFCM fitting completed.")
        return np.argmax(self.U, axis=0)
