import numpy as np
from typing import Tuple
from typeguard import typechecked


@typechecked
class CAFHFCM:
    def __init__(self, c: int, m: float = 2.0, alpha: float = 0.1,
                 max_iter: int = 100, tol: float = 1e-5):
        """
        Parameters:
        - c (int): Number of clusters
        - m (float): Fuzziness coefficient
        - alpha (float): Fusion regularization parameter
        - max_iter (int): Maximum iterations
        - tol (float): Convergence tolerance
        """
        self.c = c
        self.m = m
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.U = None

    def _update_U(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2) + 1e-8
        exponent = 2.0 / (self.m - 1.0)
        inv_dist = dist ** (-exponent)
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        return U

    def _update_centroids(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        um = U ** self.m
        base = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

        fusion_term = np.zeros_like(base)
        for i in range(self.c):
            for j in range(self.c):
                if i != j:
                    fusion_term[i] += base[i] - base[j]

        new_centroids = base - self.alpha * fusion_term
        return new_centroids

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform CAF-HFCM clustering.

        Parameters:
        - X (np.ndarray): Input data (N x D)

        Returns:
        - labels (np.ndarray): Hard cluster assignments (N,)
        - U (np.ndarray): Membership matrix (N x C)
        """
        N, D = X.shape
        self.centroids = X[np.random.choice(N, self.c, replace=False)]

        for _ in range(self.max_iter):
            U_new = self._update_U(X, self.centroids)
            centroids_new = self._update_centroids(X, U_new)

            if np.linalg.norm(self.centroids - centroids_new) < self.tol:
                break

            self.centroids = centroids_new
            self.U = U_new

        labels = np.argmax(self.U, axis=1)
        return labels, self.U
