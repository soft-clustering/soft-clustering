import numpy as np
from typing import Tuple
from typeguard import typechecked


@typechecked
class FCC:
    def __init__(self, c: int, jnd: float = 20.0, max_iter: int = 100, tol: float = 1e-5):
        """
        Parameters:
        - c (int): Number of clusters
        - jnd (float): Just Noticeable Difference (radius of fuzzy color sphere)
        - max_iter (int): Maximum number of iterations
        - tol (float): Convergence threshold
        """
        self.c = c
        self.jnd = jnd
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.U = None

    def _fuzzy_membership(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        memberships = np.zeros(self.c)
        distances = np.linalg.norm(centroids - x, axis=1)

        for i in range(self.c):
            d = distances[i]
            if d <= self.jnd:
                memberships[i] = 1.0
            else:
                memberships[i] = 1.0 / (1.0 + (d - self.jnd))

        return memberships / memberships.sum()

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Fuzzy Color Clustering.

        Parameters:
        - X (np.ndarray): Color data in CIELAB space (N x 3)

        Returns:
        - labels (np.ndarray): Hard cluster assignments (N,)
        - U (np.ndarray): Fuzzy membership matrix (N x C)
        """
        N, D = X.shape
        self.centroids = X[np.random.choice(N, self.c, replace=False)]
        self.U = np.zeros((N, self.c))

        for iteration in range(self.max_iter):
            for i in range(N):
                self.U[i] = self._fuzzy_membership(X[i], self.centroids)

            new_centroids = np.zeros((self.c, D))
            for j in range(self.c):
                weights = self.U[:, j] ** 2
                weighted_sum = np.dot(weights, X)
                total_weight = np.sum(weights)
                new_centroids[j] = weighted_sum / (total_weight + 1e-8)

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        labels = np.argmax(self.U, axis=1)
        return labels, self.U
