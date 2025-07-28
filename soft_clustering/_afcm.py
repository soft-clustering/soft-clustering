import numpy as np
from typing import Tuple
from typeguard import typechecked
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh


@typechecked
class AFCM:
    def __init__(self, c: int, lambda_: float = 1.0, m: float = 2.0, max_iter: int = 100, tol: float = 1e-5, n_neighbors: int = 5) -> None:
        """
        Parameters:
        - c (int): Number of clusters
        - lambda_ (float): Regularization parameter for graph embedding
        - m (float): Fuzziness degree
        - max_iter (int): Maximum number of iterations
        - tol (float): Tolerance for convergence
        - n_neighbors (int): Number of neighbors for kNN graph
        """
        self.c = c
        self.lambda_ = lambda_
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors = n_neighbors

    def _compute_laplacian(self, X: np.ndarray) -> np.ndarray:
        W = kneighbors_graph(X, self.n_neighbors,
                             mode='connectivity', include_self=True).toarray()
        D = np.diag(W.sum(axis=1))
        L = D - W
        return L

    def _graph_embedding(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        L = self._compute_laplacian(X)
        B = np.eye(N) - U @ np.linalg.pinv(U.T @ U) @ U.T
        M = L + self.lambda_ * B
        eigvals, eigvecs = eigh(M, eigvals=(0, self.c - 1))
        return eigvecs

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full AFCM with graph embedding.

        Parameters:
        - X (np.ndarray): Input data (N x D)

        Returns:
        - labels (np.ndarray): Cluster labels
        - U (np.ndarray): Membership matrix
        """
        N, D = X.shape
        U = np.random.dirichlet(np.ones(self.c), size=N)
        V = np.zeros((self.c, self.c))  # embedding space dimension = c

        for iteration in range(self.max_iter):
            X_tilde = self._graph_embedding(X, U)

            # Update cluster centers
            um = U ** self.m
            V = (um.T @ X_tilde) / np.sum(um.T, axis=1, keepdims=True)

            # Update distances
            dist = np.zeros((N, self.c))
            for i in range(self.c):
                dist[:, i] = np.linalg.norm(X_tilde - V[i], axis=1) + 1e-8

            # Update memberships
            tmp = dist ** (2 / (self.m - 1))
            denom = np.sum((1 / tmp), axis=1, keepdims=True)
            U_new = (1 / tmp) / denom

            if np.linalg.norm(U_new - U) < self.tol:
                break
            U = U_new

        labels = np.argmax(U, axis=1)
        return labels, U
