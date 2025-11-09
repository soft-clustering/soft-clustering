import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.ndimage import uniform_filter
from typeguard import typechecked


@typechecked
class SKFCM:
    def __init__(self, n_clusters: int = 3, m: float = 2.0, gamma: float = 1.0,
                 lambda_: float = 0.5, max_iter: int = 100, tol: float = 1e-5):
        """
        Parameters:
        - n_clusters (int): Number of clusters
        - m (float): Fuzziness degree
        - gamma (float): Kernel RBF parameter
        - lambda_ (float): Spatial constraint weight
        - max_iter (int): Max number of iterations
        - tol (float): Convergence threshold
        """
        self.n_clusters = n_clusters
        self.m = m
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.U = None
        self.K = None
        self.labels_ = None
        self.N = None

    def _initialize_U(self):
        self.U = np.random.dirichlet(np.ones(self.n_clusters), size=self.N)

    def _compute_kernel(self, X: np.ndarray):
        return rbf_kernel(X, gamma=self.gamma)

    def _spatial_term(self, U: np.ndarray, shape: Tuple[int, int]):
        spatial_U = U.copy()
        for k in range(self.n_clusters):
            u_k = U[:, k].reshape(shape)
            spatial_U[:, k] = uniform_filter(u_k, size=3).reshape(-1)
        return spatial_U

    def _update_U(self, spatial_U: np.ndarray):
        Um = self.U ** self.m
        d = np.zeros((self.N, self.n_clusters))

        for k in range(self.n_clusters):
            num = np.diag(self.K) \
                - (2 / np.sum(Um[:, k])) * (self.K @ Um[:, k]) \
                + (1 / np.sum(Um[:, k]) ** 2) * \
                (Um[:, k].T @ self.K @ Um[:, k])
            d[:, k] = num + self.lambda_ * (1 - spatial_U[:, k])

        d = np.clip(d, 1e-10, None)
        for i in range(self.N):
            denom = np.sum((d[i, :] / d[i, :][:, None])
                           ** (1 / (self.m - 1)), axis=0)
            self.U[i, :] = 1.0 / denom

    def fit(self, X: np.ndarray, shape: Tuple[int, int]):
        self.N = X.shape[0]
        self.K = self._compute_kernel(X)
        self._initialize_U()

        for _ in range(self.max_iter):
            U_old = self.U.copy()
            spatial_U = self._spatial_term(self.U, shape)
            self._update_U(spatial_U)
            if np.linalg.norm(self.U - U_old) < self.tol:
                break

        self.labels_ = np.argmax(self.U, axis=1)

    def predict(self) -> np.ndarray:
        return self.labels_

    def predict_proba(self) -> np.ndarray:
        return self.U
