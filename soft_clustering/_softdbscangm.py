"""Soft DBSCAN with Gaussian-mixture refinement.

Implementation note (optimization study)
----------------------------------------
This module is a vectorised implementation of the same algorithm. The
objective, update rules, initialisation, clamping and convergence test are
unchanged; only how they are evaluated differs. The reference implementation is
preserved at ``optimization/original/scpp_original/_softdbscangm.py`` and the
two are compared in ``tests/test_optimization_equivalence.py``.

The membership update was the bottleneck. It evaluated

    U[i, j] = 1 / sum_t ( d(i, j) / d(i, t) ) ** (2 / (m - 1))

with a Python triple loop calling ``scipy.spatial.distance.mahalanobis`` once
per (i, j, t) — N * k**2 scalar SciPy calls per iteration, recomputing each of
the N * k distances k times. Because every DBSCAN noise point becomes its own
component, k grows with N, so the cost was cubic in the sample count:
profiling measured 1,036,800 ``mahalanobis`` calls for a 80-sample input.

Two changes remove it, both algebraic identities rather than approximations:

1. The N * k Mahalanobis distances are computed once per iteration, one
   ``einsum`` per component over the whole sample matrix.
2. The ratio sum is rewritten in the equivalent closed form

       U[i, j] = d(i, j) ** -p / sum_t d(i, t) ** -p,    p = 2 / (m - 1)

   which needs no (N, k, k) tensor. Distances are divided by their per-sample
   minimum before exponentiation, so the powers stay in [0, 1] and cannot
   overflow for small ``m``; that rescaling cancels exactly between numerator
   and denominator.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typeguard import typechecked

from ._base import BaseSoftClusterer


@typechecked
class SoftDBSCANGM(BaseSoftClusterer):
    # density-based degrees are not normalised across components
    _partition_constrained = False
    # Discovers the cluster count; see BaseSoftClusterer._determines_k.
    _determines_k = True

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        """
        Parameters:
        - eps (float): DBSCAN epsilon radius
        - min_samples (int): Minimum number of neighbors for DBSCAN
        - m (float): Fuzziness degree
        - max_iter (int): Max iterations for fuzzy refinement
        - tol (float): Convergence tolerance
        """
        self.eps = eps
        self.min_samples = min_samples
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.U = None
        self.centers = None
        self.cov_inv = None

    @staticmethod
    def _chunk_rows(n_samples: int, k: int, n_features: int) -> int:
        """Rows per block, so the (rows, k, d) temporary stays near 32 MB.

        Every DBSCAN noise point becomes its own component, so k grows with the
        sample count and a full (n, k, d) array would be quadratic in memory.
        Blocking keeps the working set bounded while still handing BLAS a batch
        large enough to amortise the call.
        """
        per_row = max(1, k * n_features)
        return int(np.clip(4_000_000 // per_row, 1, n_samples))

    def _covariances(self, X: np.ndarray, U_m: np.ndarray, weights) -> np.ndarray:
        """Membership-weighted covariance of every component, shape (k, d, d)."""
        n_samples, n_features = X.shape
        k = self.centers.shape[0]
        cov = np.zeros((k, n_features, n_features))
        step = self._chunk_rows(n_samples, k, n_features)
        for start in range(0, n_samples, step):
            stop = start + step
            diff = X[start:stop, None, :] - self.centers[None, :, :]
            cov += np.einsum(
                "nk,nkd,nke->kde", U_m[start:stop], diff, diff, optimize=True
            )
        return cov / (weights[:, None, None] + 1e-10)

    def _mahalanobis_distances(self, X: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
        """All (n_samples, k) Mahalanobis distances to the current centres.

        Batched over components: one ``einsum`` per block of rows contracts
        ``diff @ cov_inv @ diff`` for every component at once, replacing the
        ``n_samples * k`` scalar SciPy calls of the reference.
        """
        n_samples, n_features = X.shape
        k = self.centers.shape[0]
        d2 = np.empty((n_samples, k))
        step = self._chunk_rows(n_samples, k, n_features)
        for start in range(0, n_samples, step):
            stop = start + step
            diff = X[start:stop, None, :] - self.centers[None, :, :]
            d2[start:stop] = np.einsum(
                "nkd,kde,nke->nk", diff, cov_inv, diff, optimize=True
            )
        # Rounding can put a near-zero quadratic form marginally below zero.
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    def fit(self, X: np.ndarray):
        N, D = X.shape

        # Step 1: DBSCAN clustering
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        raw_labels = db.labels_

        # Unique labels, -1 = noise
        cluster_ids = np.unique(raw_labels)
        cluster_ids = cluster_ids[cluster_ids != -1]
        noise_ids = np.where(raw_labels == -1)[0]

        k = len(cluster_ids) + len(noise_ids)
        self.U = np.zeros((N, k))

        # Step 2: initialise the membership matrix. Each DBSCAN cluster owns a
        # column; each noise point owns a column of its own, in ascending index
        # order. The reference did this with list.index() inside a loop over
        # samples, which is quadratic; searchsorted gives the same columns.
        is_noise = raw_labels == -1
        columns = np.empty(N, dtype=np.intp)
        columns[is_noise] = len(cluster_ids) + np.arange(len(noise_ids))
        columns[~is_noise] = np.searchsorted(cluster_ids, raw_labels[~is_noise])
        self.U[np.arange(N), columns] = 1.0

        self.centers = np.zeros((k, D))
        self.cov_inv = [np.eye(D)] * k

        exponent = 2.0 / (self.m - 1.0)

        for _iteration in range(self.max_iter):
            U_m = self.U**self.m
            prev_centers = self.centers.copy()

            # Step 3: update centers. One GEMM for all components at once.
            weights = U_m.sum(axis=0)
            self.centers = (U_m.T @ X) / (weights[:, None] + 1e-10)

            # Step 4: update covariance inverses. One batched LAPACK call for
            # all k components instead of k separate inversions.
            cov = self._covariances(X, U_m, weights)
            cov += np.eye(D) * 1e-6
            cov_inv = np.linalg.inv(cov)
            # Preserved as a list of (d, d) arrays, as the reference exposed it.
            self.cov_inv = list(cov_inv)

            # Step 5: update memberships.
            distances = np.maximum(self._mahalanobis_distances(X, cov_inv), 1e-10)
            # Rescale by the per-sample minimum before exponentiating: the
            # factor cancels between numerator and denominator, and keeps
            # every power within [0, 1] regardless of how small m - 1 is.
            scaled = distances / distances.min(axis=1, keepdims=True)
            weights_inv = scaled**-exponent
            self.U = weights_inv / weights_inv.sum(axis=1, keepdims=True)

            # Check convergence
            if np.linalg.norm(self.centers - prev_centers) < self.tol:
                break

        self.labels_ = np.argmax(self.U, axis=1)

    def get_membership(self) -> np.ndarray:
        return self.U

    def predict(self) -> np.ndarray:
        self._check_fitted()
        return self.labels_
