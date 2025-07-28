import numpy as np
from typing import Tuple, List
from typeguard import typechecked


@typechecked
class AFCMSimple:
    def __init__(self, c: int, m: float = 2.0, max_iter: int = 100, tol: float = 1e-5) -> None:
        """
        Parameters:
        - c (int): Number of clusters
        - m (float): Fuzziness degree
        - max_iter (int): Maximum number of iterations
        - tol (float): Tolerance for convergence
        """
        self.c = c
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Adaptive FCM without graph embedding.

        Parameters:
        - X (np.ndarray): Input data (N x D)

        Returns:
        - labels (np.ndarray): Cluster labels (length N)
        - U (np.ndarray): Membership matrix (N x C)
        """
        N, D = X.shape
        U = np.random.dirichlet(np.ones(self.c), size=N)
        V = np.zeros((self.c, D))

        for iteration in range(self.max_iter):
            # Update cluster centers
            um = U ** self.m
            V = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

            # Update distances
            dist = np.zeros((N, self.c))
            for i in range(self.c):
                dist[:, i] = np.linalg.norm(X - V[i], axis=1) + 1e-8

            # Update memberships
            tmp = dist ** (2 / (self.m - 1))
            denom = np.sum((1 / tmp), axis=1, keepdims=True)
            U_new = (1 / tmp) / denom

            # Check convergence
            if np.linalg.norm(U_new - U) < self.tol:
                break
            U = U_new

        labels = np.argmax(U, axis=1)
        return labels, U
