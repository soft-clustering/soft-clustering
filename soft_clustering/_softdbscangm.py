import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from typeguard import typechecked


@typechecked
class SoftDBSCANGM:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, m: float = 2.0,
                 max_iter: int = 100, tol: float = 1e-4):
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

        # Step 2: Initialize membership matrix
        for i in range(N):
            if raw_labels[i] == -1:
                self.U[i, len(cluster_ids) + list(noise_ids).index(i)] = 1.0
            else:
                cluster_idx = list(cluster_ids).index(raw_labels[i])
                self.U[i, cluster_idx] = 1.0

        self.centers = np.zeros((k, D))
        self.cov_inv = [np.eye(D)] * k

        for iteration in range(self.max_iter):
            U_m = self.U ** self.m
            prev_centers = self.centers.copy()

            # Step 3: update centers
            for j in range(k):
                numerator = np.dot(U_m[:, j], X)
                denominator = np.sum(U_m[:, j])
                self.centers[j] = numerator / (denominator + 1e-10)

            # Step 4: update covariance inverses
            for j in range(k):
                diff = X - self.centers[j]
                weighted = (U_m[:, j][:, None] * diff)
                cov = np.dot(weighted.T, diff) / (np.sum(U_m[:, j]) + 1e-10)
                self.cov_inv[j] = inv(cov + np.eye(D) * 1e-6)

            # Step 5: update memberships
            for i in range(N):
                for j in range(k):
                    d = mahalanobis(X[i], self.centers[j], self.cov_inv[j])
                    d = max(d, 1e-10)
                    denom = sum(
                        (d / max(mahalanobis(X[i], self.centers[t],
                         self.cov_inv[t]), 1e-10)) ** (2 / (self.m - 1))
                        for t in range(k)
                    )
                    self.U[i, j] = 1.0 / denom

            # Check convergence
            if np.linalg.norm(self.centers - prev_centers) < self.tol:
                break

        self.labels_ = np.argmax(self.U, axis=1)

    def get_membership(self) -> np.ndarray:
        return self.U

    def predict(self) -> np.ndarray:
        return self.labels_
