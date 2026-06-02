import numpy as np
from typing import Tuple
from typeguard import typechecked


@typechecked
class ENTROPYFCM:
    def __init__(self, c: int, m: float = 2.0, entropy_weight: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-5) -> None:
        """
        Parameters:
        - c (int): Number of clusters
        - m (float): Fuzziness degree
        - entropy_weight (float): Weighting factor for entropy objective
        - max_iter (int): Maximum number of iterations
        - tol (float): Convergence tolerance
        """
        self.c = c
        self.m = m
        self.entropy_weight = entropy_weight
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.U = None

    def _initialize(self, X: np.ndarray) -> None:
        N = X.shape[0]
        self.U = np.random.dirichlet(np.ones(self.c), size=N)
        self.centroids = X[np.random.choice(N, self.c, replace=False)]

    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        um = self.U ** self.m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

    def _update_memberships(self, X: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(
            X[:, None, :] - self.centroids[None, :, :], axis=2) + 1e-8
        inv_dist = dist ** (-2 / (self.m - 1))
        return inv_dist / np.sum(inv_dist, axis=1, keepdims=True)

    def _objective(self, X: np.ndarray) -> Tuple[float, float]:
        um = self.U ** self.m
        compactness = np.sum(
            um * np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2) ** 2)
        entropy = -np.sum(self.U * np.log(self.U + 1e-8))
        return compactness, entropy

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run ECM clustering (base mode, single run).

        Parameters:
        - X (np.ndarray): Input data (N x D)

        Returns:
        - labels (np.ndarray): Hard cluster assignments
        - U (np.ndarray): Membership matrix (N x C)
        """
        self._initialize(X)

        for _ in range(self.max_iter):
            centroids_new = self._update_centroids(X)
            if np.linalg.norm(self.centroids - centroids_new) < self.tol:
                break
            self.centroids = centroids_new
            self.U = self._update_memberships(X)

        labels = np.argmax(self.U, axis=1)
        return labels, self.U
