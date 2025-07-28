import numpy as np
from typing import Tuple
from typeguard import typechecked


@typechecked
class RPFKM:
    def __init__(self, c: int, d: int, gamma: float = 0.1, beta: float = 1.0, max_iter: int = 50) -> None:
        """
        Initializes the RPFKM algorithm with given hyperparameters.

        Parameters:
        - c (int): Number of clusters
        - d (int): Reduced dimension
        - gamma (float): Regularization for fuzzy membership
        - beta (float): Regularization for data reconstruction
        - max_iter (int): Number of iterations
        """
        self.c = c
        self.d = d
        self.gamma = gamma
        self.beta = beta
        self.max_iter = max_iter

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the RPFKM algorithm on the given data matrix X.

        Parameters:
        - X (np.ndarray): Data matrix of shape (D, N)

        Returns:
        - cluster_labels (np.ndarray): Final cluster assignments
        - U (np.ndarray): Fuzzy membership matrix
        - W (np.ndarray): Learned projection matrix
        """
        W, U, p = self._initialize_variables(X)
        for _ in range(self.max_iter):
            M = self._update_M(X, W, U, p)
            p = self._update_p(X, W, M)
            U = self._update_U(X, W, M)
            W = self._update_W(X, U, p)
        cluster_labels = np.argmax(U, axis=0)
        return cluster_labels, U, W

    def _initialize_variables(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D, N = X.shape
        np.random.seed(0)
        W = np.linalg.qr(np.random.randn(D, self.d))[0]
        U = np.random.dirichlet(np.ones(self.c), size=N).T
        p = np.ones((self.c, N))
        return W, U, p

    def _update_M(self, X: np.ndarray, W: np.ndarray, U: np.ndarray, p: np.ndarray) -> np.ndarray:
        WX = W.T @ X
        M = np.zeros((self.d, self.c))
        for k in range(self.c):
            numerator = np.sum((p[k, :] * U[k, :]) * WX, axis=1)
            denominator = np.sum(p[k, :] * U[k, :]) + 1e-10
            M[:, k] = numerator / denominator
        return M

    def _update_p(self, X: np.ndarray, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        WX = W.T @ X
        p = np.zeros((self.c, X.shape[1]))
        for k in range(self.c):
            diff = WX - M[:, k].reshape(-1, 1)
            norm = np.linalg.norm(diff, axis=0) + 1e-8
            p[k, :] = 1.0 / (2.0 * norm)
        return p

    def _update_U(self, X: np.ndarray, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        WX = W.T @ X
        f = np.zeros((self.c, WX.shape[1]))
        for k in range(self.c):
            diff = WX - M[:, k].reshape(-1, 1)
            f[k, :] = np.linalg.norm(diff, axis=0)
        u = -f / (2 * self.gamma)
        u = np.exp(u)
        u /= np.sum(u, axis=0, keepdims=True)
        return u

    def _update_W(self, X: np.ndarray, U: np.ndarray, p: np.ndarray) -> np.ndarray:
        D, N = X.shape
        X_mean = X @ np.ones((N, 1)) / N
        Xt = X - X_mean
        S_w = np.zeros((D, D))
        for k in range(self.c):
            s = (p[k, :] * U[k, :])
            x_mean_k = np.sum(s * X, axis=1, keepdims=True) / np.sum(s)
            diff = X - x_mean_k
            S_w += diff @ np.diag(s) @ diff.T
        S_t = Xt @ Xt.T
        eigvals, eigvecs = np.linalg.eigh(self.beta * S_t - S_w)
        W = eigvecs[:, -self.d:]
        return W
