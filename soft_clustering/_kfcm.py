import numpy as np
from typing import Optional
from typeguard import typechecked

@typechecked
class KFCM:
    """
    An improved Python implementation of the Kernelized Fuzzy C-Means (KFCM) algorithm.
    
    This version incorporates K-Means++ initialization for more robust and consistent
    clustering results, addressing the issue of poor random starts.
    """
    def __init__(self, n_clusters: int = 3,
                 m: float = 2.0,
                 sigma: float = 1.0,
                 epsilon: float = 0.01,
                 max_iter: int = 100
                 ):
        """
        Initializes the KFCM algorithm with given parameters.
        """
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if m <= 1.0:
            raise ValueError("Fuzziness exponent m must be > 1.")
        if sigma <= 0:
            raise ValueError("Sigma must be a positive float.")
        
        self.n_clusters: int = n_clusters
        self.m: float = m
        self.sigma: float = sigma
        self.epsilon: float = epsilon
        self.max_iter: int = max_iter
        self.V: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None

    def _initialize_centers_kmeans_pp(self, X: np.ndarray) -> np.ndarray:
        """
        Initializes cluster centers using the K-Means++ strategy.
        This method spreads out the initial centers, leading to better convergence.
        """
        N, D = X.shape
        centers = np.zeros((self.n_clusters, D))
        
        # 1. Choose the first center uniformly at random from the data points
        first_center_idx = np.random.randint(N)
        centers[0] = X[first_center_idx]
        
        # 2. For the remaining centers
        for i in range(1, self.n_clusters):
            # Calculate the squared distance of each point to the nearest already-chosen center
            dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in centers[:i]]) for x in X])
            
            # 3. Choose the next center with probability proportional to the squared distance
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centers[i] = X[j]
                    break
        return centers

    def _gaussian_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Gaussian RBF kernel between two points.
        """
        return float(np.exp(-np.linalg.norm(x - y)**2 / self.sigma**2))

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the KFCM model to the data X.
        """
        N, D = X.shape
        if N == 0:
            raise ValueError("Input data cannot be empty.")

        # --- Step 1: Initialization ---
        self.V = self._initialize_centers_kmeans_pp(X).astype(np.float64)
        
        self.U = np.random.rand(self.n_clusters, N)
        
        self.U = self.U / np.sum(self.U, axis=0)
        
        # --- Step 2: The Iteration Loop ---
        for t in range(self.max_iter):
            U_old = self.U.copy()

            # --- Step 3: Update Cluster Centers (V) ---
            for i in range(self.n_clusters):
                K_vi = np.array([self._gaussian_kernel(x_k, self.V[i, :]) for x_k in X]) 
                numerator = np.sum((self.U[i, :]**self.m * K_vi)[:, np.newaxis] * X, axis=0)
                denominator = np.sum(self.U[i, :]**self.m * K_vi)
                
                if denominator > 1e-9:
                    self.V[i, :] = numerator / denominator

            # --- Step 4: Update Membership Matrix (U) ---
            for k in range(N):
                dist_k = np.array([1 - self._gaussian_kernel(X[k, :], self.V[i, :]) for i in range(self.n_clusters)])
                dist_k[dist_k == 0] = np.finfo(float).eps
                denominator = np.sum((1 / dist_k)**(1 / (self.m - 1)))
                
                if denominator > 1e-9:
                    self.U[:, k] = ((1 / dist_k)**(1 / (self.m - 1))) / denominator
            
            # --- Check for convergence ---
            if np.max(np.abs(self.U - U_old)) < self.epsilon:
                print(f"Converged at iteration {t+1}")
                break
        
        if self.U is None:
             raise RuntimeError("Fitting failed, membership matrix is None.")

        print("KFCM fitting completed.")
        return np.argmax(self.U, axis=0)
