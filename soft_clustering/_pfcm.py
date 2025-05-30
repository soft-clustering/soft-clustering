import numpy as np
from typeguard import typechecked
from typing import Union, Optional, List

@typechecked
class PFCM:
    def __init__(
        self,
        n_clusters: int,             # Number of clusters
        membership_fuzzifier: float = 2.0,              # Fuzzifier for memberships (typically m > 1)
        typicality_fuzzifier: float = 2.0,            # Fuzzifier for typicalities
        membership_weight: float = 1.0,              # Weight for membership term
        typicality_weight: float = 1.0,              # Weight for typicality term
        max_iter: int = 150,         # Maximum number of iterations
        tol: float = 1e-5,           # Tolerance for convergence
        random_state: Optional[int] = None  # Random seed for reproducibility
    ):
        # Store parameters
        self.n_clusters = n_clusters
        self.m = membership_fuzzifier
        self.eta = typicality_fuzzifier
        self.a = membership_weight
        self.b = typicality_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Will be set during training
        self.membership_matrix: Optional[np.ndarray] = None     # Membership matrix (c x n)
        self.typicality_matrix: Optional[np.ndarray] = None     # Typicality matrix (c x n)
        self.cluster_centroids: Optional[np.ndarray] = None     # Cluster centroids (c x d)
        self.gamma: Optional[np.ndarray] = None # Cluster-wise scaling for typicalities

    def _initialize_centroids(self, X: np.ndarray) -> None:
        """
        Initialize cluster centers by selecting c random points from X.
        """
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centroids = X[indices]

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances between each point and cluster center.
        Returns a (c x n) distance matrix.
        """
        distances = np.zeros((self.n_clusters, X.shape[0]))
        for i, v in enumerate(self.cluster_centroids):
            distances[i] = np.linalg.norm(X - v, axis=1) ** 2 + 1e-10  # Add small constant to avoid division by zero
        return distances

    def _update_memberships(self, distances: np.ndarray) -> None:
        """
        Update fuzzy membership matrix using the standard FCM formula.
        Each column in membership_matrix sums to 1.
        """
        power = 1.0 / (self.m - 1)
        denom = np.sum((distances[:, None, :] / distances[None, :, :]) ** power, axis=1)
        self.membership_matrix = 1.0 / denom

    def _update_gamma(self, distances: np.ndarray) -> None:
        """
        Compute gamma_i for each cluster, used to scale the typicality update.
        This is based on the average intra-cluster distance.
        """
        self.gamma = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            num = np.sum((self.membership_matrix[i] ** self.m) * distances[i])
            den = np.sum(self.membership_matrix[i] ** self.m)
            self.gamma[i] = num / den

    def _update_typicalities(self, distances: np.ndarray) -> None:
        """
        Update typicality matrix based on distance and gamma.
        Lower distances => higher typicality.
        """
        self.typicality_matrix = np.zeros_like(self.membership_matrix)
        for i in range(self.n_clusters):
            ratio = distances[i] / self.gamma[i]
            self.typicality_matrix[i] = 1.0 / (1.0 + ratio ** (1.0 / (self.eta - 1)))

    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Update cluster centers using both memberships and typicalities.
        Each point contributes weighted by a * u^m + b * t^eta.
        """
        new_cluster_centroids = np.zeros_like(self.cluster_centroids)
        for i in range(self.n_clusters):
            u_term = self.a * (self.membership_matrix[i] ** self.m)
            t_term = self.b * (self.typicality_matrix[i] ** self.eta)
            weights = u_term + t_term
            weighted_sum = np.dot(weights, X)             # Weighted average of points
            new_cluster_centroids[i] = weighted_sum / np.sum(weights)     # New centroid
        return new_cluster_centroids

    def fit(self, X: Union[np.ndarray, List[List[float]]]) -> "PFCM":
        """
        Fit the PFCM model to data X.
        Returns self with updated membership_matrix, typicality_matrix, cluster_centroids.
        """
        X = np.array(X, dtype=np.float64)
        self._initialize_centroids(X)

        for _ in range(self.max_iter):
            distances = self._compute_distances(X)
            self._update_memberships(distances)
            self._update_gamma(distances)
            self._update_typicalities(distances)
            new_cluster_centroids = self._update_centroids(X)

            # Convergence: check max L2 change in centroids
            max_change = np.max(np.linalg.norm(self.cluster_centroids - new_cluster_centroids, axis=1))
            self.cluster_centroids = new_cluster_centroids
            if max_change < self.tol:
                break

        return self

    def predict_memberships(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Predict fuzzy memberships for new data points.
        Returns a (c x n_new) membership matrix.
        """
        X = np.array(X, dtype=np.float64)
        distances = self._compute_distances(X)
        power = 1.0 / (self.m - 1)
        denom = np.sum((distances[:, None, :] / distances[None, :, :]) ** power, axis=1)
        return 1.0 / denom

    def predict_typicalities(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Predict typicality values for new data points.
        Returns a (c x n_new) typicality matrix.
        """
        X = np.array(X, dtype=np.float64)
        distances = self._compute_distances(X)
        T = np.zeros((self.n_clusters, X.shape[0]))
        for i in range(self.n_clusters):
            ratio = distances[i] / self.gamma[i]
            T[i] = 1.0 / (1.0 + ratio ** (1.0 / (self.eta - 1)))
        return T