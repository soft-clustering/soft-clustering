import numpy as np
from typing import Tuple
from typeguard import typechecked


@typechecked
class CAFCM:
    def __init__(self, c: int, m_start: float = 2.0, m_end: float = 1.01,
                 cooling_rate: float = 0.95, max_iter: int = 100, tol: float = 1e-5):
        """
        Parameters:
        - c (int): Number of clusters
        - m_start (float): Initial fuzziness degree
        - m_end (float): Final fuzziness degree (close to 1)
        - cooling_rate (float): Annealing rate to reduce m
        - max_iter (int): Maximum number of iterations per m value
        - tol (float): Convergence tolerance
        """
        self.c = c
        self.m_start = m_start
        self.m_end = m_end
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.U = None

    def _update_U(self, X: np.ndarray, centroids: np.ndarray, m: float) -> np.ndarray:
        dist = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2) + 1e-8
        exponent = 2.0 / (m - 1.0)
        inv_dist = dist ** (-exponent)
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        return U

    def _update_centroids(self, X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
        um = U ** m
        centroids = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
        return centroids

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform collaborative annealing fuzzy C-means clustering.

        Parameters:
        - X (np.ndarray): Input data (N x D)

        Returns:
        - labels (np.ndarray): Final hard labels
        - U (np.ndarray): Final membership matrix
        """
        N, D = X.shape
        m = self.m_start
        self.centroids = X[np.random.choice(N, self.c, replace=False)]

        while m > self.m_end:
            for iteration in range(self.max_iter):
                U_new = self._update_U(X, self.centroids, m)
                centroids_new = self._update_centroids(X, U_new, m)

                if np.linalg.norm(self.centroids - centroids_new) < self.tol:
                    break

                self.centroids = centroids_new
                self.U = U_new

            m *= self.cooling_rate  # annealing step

        labels = np.argmax(self.U, axis=1)
        return labels, self.U
