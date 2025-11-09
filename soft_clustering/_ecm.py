import numpy as np
from typeguard import typechecked


@typechecked
class ECM:
    def __init__(self, n_clusters: int = 3, m: float = 2.0, delta: float = 10.0, max_iter: int = 100, tol: float = 1e-5):
        """
        Parameters:
        - n_clusters (int): Number of clusters
        - m (float): Fuzziness parameter
        - delta (float): Distance threshold for the noise cluster
        - max_iter (int): Maximum number of iterations
        - tol (float): Convergence threshold
        """
        self.n_clusters = n_clusters
        self.m = m
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        self.prototypes = None
        self.mass = None

    def _initialize_prototypes(self, X: np.ndarray):
        N, _ = X.shape
        indices = np.random.choice(N, self.n_clusters, replace=False)
        self.prototypes = X[indices]

    def _compute_distances(self, X: np.ndarray):
        N = X.shape[0]
        dist = np.zeros((N, self.n_clusters))
        for k in range(self.n_clusters):
            dist[:, k] = np.linalg.norm(X - self.prototypes[k], axis=1) ** 2
        return dist

    def fit(self, X: np.ndarray):
        N = X.shape[0]
        self._initialize_prototypes(X)

        for _ in range(self.max_iter):
            D = self._compute_distances(X)
            D = np.clip(D, 1e-10, None)  # Avoid division by zero
            m1 = 1. / (self.m - 1)

            # Last column = noise cluster
            M = np.zeros((N, self.n_clusters + 1))

            for i in range(N):
                for k in range(self.n_clusters):
                    M[i, k] = (1.0 / D[i, k]) ** m1
                M[i, -1] = (1.0 / self.delta ** 2) ** m1  # Noise mass
                M[i, :] = M[i, :] / np.sum(M[i, :])

            self.mass = M

            new_prototypes = np.zeros_like(self.prototypes)
            for k in range(self.n_clusters):
                weights = M[:, k] ** self.m
                numerator = np.dot(weights, X)
                denominator = np.sum(weights)
                new_prototypes[k] = numerator / (denominator + 1e-10)

            if np.linalg.norm(new_prototypes - self.prototypes) < self.tol:
                break
            self.prototypes = new_prototypes

    def get_membership(self) -> np.ndarray:
        """
        Returns:
        - mass (np.ndarray): Mass matrix (n_samples x n_clusters + 1)
        """
        return self.mass
