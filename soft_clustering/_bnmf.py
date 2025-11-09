import numpy as np
from typeguard import typechecked


@typechecked
class BNMF:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, a: float = 1.0, b: float = 1.0, tol: float = 1e-5):
        """
        Parameters:
        - n_clusters (int): Number of latent communities (K)
        - max_iter (int): Maximum number of iterations
        - a (float): Gamma prior shape parameter
        - b (float): Gamma prior rate parameter
        - tol (float): Convergence threshold
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.a = a
        self.b = b
        self.tol = tol
        self.W = None
        self.H = None
        self.beta = None

    def fit(self, V: np.ndarray):
        N = V.shape[0]
        K = self.n_clusters

        # Initialize W, H with small random positive values
        W = np.random.rand(N, K)
        H = np.random.rand(K, N)
        beta = np.ones(K)

        for _ in range(self.max_iter):
            WH = np.dot(W, H) + 1e-10
            V_div_WH = V / WH

            # Update H
            numerator_H = np.dot(W.T, V_div_WH)
            denominator_H = np.dot(W.T, np.ones((N, N))) + \
                H * beta[:, np.newaxis]
            H *= numerator_H / (denominator_H + 1e-10)

            # Update W
            WH = np.dot(W, H) + 1e-10
            V_div_WH = V / WH
            numerator_W = np.dot(V_div_WH, H.T)
            denominator_W = np.dot(np.ones((N, N)), H.T) + \
                W * beta[np.newaxis, :]
            W *= numerator_W / (denominator_W + 1e-10)

            # Update beta
            for k in range(K):
                norm_w = np.sum(W[:, k] ** 2)
                norm_h = np.sum(H[k, :] ** 2)
                beta[k] = (norm_w + norm_h) / 2 + self.b / (N + self.a - 1)

            # Check convergence (optional)
            if np.linalg.norm(np.dot(W, H) - WH) < self.tol:
                break

        self.W = W
        self.H = H
        self.beta = beta

    def get_membership(self) -> np.ndarray:
        """
        Returns:
        - W (np.ndarray): Soft membership matrix (N x K)
        """
        return self.W
