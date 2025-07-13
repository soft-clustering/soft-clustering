import numpy as np
from typing import Optional
from typeguard import typechecked


@typechecked
class KFCCL:
    """
    Kernel-based Fuzzy Competitive Learning Clustering (K-FCCL).
    """

    # Class-level type hints
    n_clusters: int
    lambda_: float
    gamma: float
    epsilon: float
    max_iter: int
    U: Optional[np.ndarray]
    p_ik: Optional[np.ndarray]
    K: Optional[np.ndarray]

    def __init__(self, 
                 n_clusters: int = 2,
                 lambda_: float = 10.0,
                 gamma: float = 1.0,
                 epsilon: float = 1e-4,
                 max_iter: int = 100):
        self.n_clusters = n_clusters  # Number of clusters
        self.lambda_ = lambda_        # Controls fuzziness level
        self.gamma = gamma            # Gaussian kernel parameter
        self.epsilon = epsilon        # Convergence threshold
        self.max_iter = max_iter      # Maximum iterations
        self.U: Optional[np.ndarray] = None   # Membership matrix
        self.p_ik: Optional[np.ndarray] = None  # Inner products for clusters
        self.K: Optional[np.ndarray] = None   # Kernel matrix

    def _gaussian_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes Gaussian RBF kernel matrix.

        Parameters:
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns:
        -------
        K : ndarray of shape (n_samples, n_samples)
        """
        sq_dists = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the model to X.

        Parameters:
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns:
        -------
        labels : ndarray of shape (n_samples,)
        """
        N: int = X.shape[0]
        self.K = self._gaussian_kernel_matrix(X)
        K_diag: np.ndarray = np.sqrt(np.diag(self.K))

        # Initialize inner products and membership matrix
        self.p_ik = np.random.rand(self.n_clusters, N) * 0.01
        self.U = np.zeros((self.n_clusters, N))
        V_sq: np.ndarray = np.ones(self.n_clusters)

        for t in range(self.max_iter):
            eta: float = 0.05 / (1 + t)
            p_old: np.ndarray = self.p_ik.copy()

            # Membership update via softmax
            exp_lambda_p = np.exp(self.lambda_ * self.p_ik)
            self.U = exp_lambda_p / np.sum(exp_lambda_p, axis=0, keepdims=True)

            # Update inner products and center norms
            for i in range(self.n_clusters):
                V_sq[i] += 2 * eta * np.sum(self.U[i] * self.p_ik[i])
                V_sq[i] += eta ** 2 * np.sum((self.U[i][:, None] * self.U[i][None, :]) * self.K)

                for k in range(N):
                    kernel_norm: np.ndarray = self.K[:, k] / (K_diag * K_diag[k])
                    self.p_ik[i, k] += eta * np.sum(self.U[i] * kernel_norm)
                    self.p_ik[i, k] /= np.sqrt(V_sq[i])

            # Convergence check
            if np.max(np.abs(self.p_ik - p_old)) < self.epsilon:
                print(f"K-FCCL converged at iteration {t+1}")
                break

        # Return hard cluster assignments (winner cluster)
        return np.argmax(self.U, axis=0)




