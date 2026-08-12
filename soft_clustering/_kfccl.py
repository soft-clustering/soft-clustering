"""Kernel-based Fuzzy Competitive Learning Clustering (K-FCCL).

Implementation note (optimization study)
----------------------------------------
Vectorised implementation of the same algorithm. The update rules, the
initialisation draws, the learning-rate schedule and the convergence test are
unchanged. The reference implementation is preserved at
``optimization/original/scpp_original/_kfccl.py``.

Profiling attributed the runtime to 41,346 ``np.sum`` calls per fit — a
per-element Python reduction. Two redundancies caused them:

1. The normalised kernel column ``K[:, k] / (K_diag * K_diag[k])`` was rebuilt
   inside the innermost loop, so an ``N``-vector division ran once per
   ``(iteration, cluster, sample)`` even though ``K`` and ``K_diag`` are fixed
   for the whole fit. It is now computed **once**, before the iteration loop,
   as the full matrix ``Knorm[j, k] = K[j, k] / (K_diag[j] * K_diag[k])``.

2. The inner-product update was written as a loop over ``k``. Every entry
   ``p_ik[i, k]`` depends only on its own previous value and on quantities that
   are constant across the loop (``U[i]``, ``Knorm``, ``V_sq[i]``, which is
   fully determined *before* the loop begins), so the loop carries no
   dependency and collapses to one matrix-vector product::

       sum_j U[i, j] * K[j, k] / (K_diag[j] * K_diag[k])  ==  (U[i] @ Knorm)[k]

The squared-norm accumulation is rewritten with the same identity in the
opposite direction: ``sum_{j,l} U[i,j] U[i,l] K[j,l]`` is the quadratic form
``U[i] @ K @ U[i]``, which avoids materialising the ``(N, N)`` outer product
``U[i][:, None] * U[i][None, :]`` once per cluster per iteration.

Both are exact rearrangements — no approximation, and no change to the order in
which ``V_sq`` and ``p_ik`` are updated relative to each other.
"""

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer


@typechecked
class KFCCL(BaseSoftClusterer):
    """
    Kernel-based Fuzzy Competitive Learning Clustering (K-FCCL).
    """

    # Class-level type hints
    n_clusters: int
    lambda_: float
    gamma: float
    epsilon: float
    max_iter: int
    U: np.ndarray | None
    p_ik: np.ndarray | None
    K: np.ndarray | None

    def __init__(
        self,
        n_clusters: int = 2,
        lambda_: float = 10.0,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
        max_iter: int = 100,
    ):
        self.n_clusters = n_clusters  # Number of clusters
        self.lambda_ = lambda_  # Controls fuzziness level
        self.gamma = gamma  # Gaussian kernel parameter
        self.epsilon = epsilon  # Convergence threshold
        self.max_iter = max_iter  # Maximum iterations
        self.U: np.ndarray | None = None  # Membership matrix
        self.p_ik: np.ndarray | None = None  # Inner products for clusters
        self.K: np.ndarray | None = None  # Kernel matrix

    def _gaussian_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes Gaussian RBF kernel matrix.

        Parameters:
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns:
        -------
        K : ndarray of shape (n_samples, n_samples)
        """
        sq_dists = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(X**2, axis=1)
            - 2 * np.dot(X, X.T)
        )
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the model to X.

        Parameters:
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns:
        -------
        labels : ndarray of shape (n_samples,)
        """
        N: int = X.shape[0]
        if self.n_clusters > N:
            raise ValueError(
                f"n_clusters={self.n_clusters} exceeds the number of samples ({N})."
            )
        self.K = self._gaussian_kernel_matrix(X)
        K_diag: np.ndarray = np.sqrt(np.diag(self.K))

        # Cosine-normalised kernel, constant for the whole fit. The reference
        # rebuilt one column of this per (iteration, cluster, sample).
        K_norm: np.ndarray = self.K / np.outer(K_diag, K_diag)

        # Initialize inner products and membership matrix
        self.p_ik = np.random.rand(self.n_clusters, N) * 0.01
        self.U = np.zeros((self.n_clusters, N))
        V_sq: np.ndarray = np.ones(self.n_clusters)

        for t in range(self.max_iter):
            eta: float = 0.05 / (1 + t)
            p_old: np.ndarray = self.p_ik.copy()

            # Membership update via softmax
            exp_lambda_p = np.exp(self.lambda_ * self.p_ik)
            self.U = exp_lambda_p / np.sum(exp_lambda_p, axis=0, keepdims=True)

            # Update inner products and center norms
            for i in range(self.n_clusters):
                V_sq[i] += 2 * eta * np.sum(self.U[i] * self.p_ik[i])
                # sum_{j,l} U[i,j] U[i,l] K[j,l] == U[i] @ K @ U[i]
                V_sq[i] += eta**2 * (self.U[i] @ self.K @ self.U[i])

                # Every k is independent of the others: V_sq[i] is already
                # final, and U[i] and K_norm do not change within the loop.
                self.p_ik[i] += eta * (self.U[i] @ K_norm)
                self.p_ik[i] /= np.sqrt(V_sq[i])

            # Convergence check
            if np.max(np.abs(self.p_ik - p_old)) < self.epsilon:
                print(f"K-FCCL converged at iteration {t+1}")
                break

        # Return hard cluster assignments (winner cluster)
        return np.argmax(self.U, axis=0)
