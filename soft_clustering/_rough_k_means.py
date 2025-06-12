import numpy as np
from typing import Dict, Any
from typeguard import typechecked

class RoughKMeans:
    @typechecked
    def __init__(
        self,
        n_clusters: int = 2,
        weight_lower: float = 0.7,
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        Rough K-Means clustering with interval-set (lower/upper) approximations.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters.
        weight_lower : float
            Mixing weight for lower vs. upper when updating centroids.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance for centroid changes.
        """
        self.n_clusters = n_clusters
        self.weight_lower = weight_lower
        self.max_iter = max_iter
        self.tol = tol

    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two points."""
        return np.linalg.norm(a - b)

    def fit(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Perform Rough K-Means clustering on the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix where each row is a sample and each column is a feature.

        Returns
        -------
        result : dict
            Dictionary containing clustering results with keys:
            - 'lower_approx' : ndarray of shape (n_samples, n_clusters)
              Binary matrix indicating certain membership (1 = in lower approximation)
            - 'upper_approx' : ndarray of shape (n_samples, n_clusters)
              Binary matrix indicating possible membership (1 = in upper approximation)
            - 'centroids' : ndarray of shape (n_clusters, n_features)
              Final cluster centroids
            - 'n_iter' : int
              Number of iterations performed
        """
        # Validate input dimensions
        n_samples, _ = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("Not enough samples for the number of clusters.")

        # Initialize centroids using random samples
        rng = np.random.default_rng()
        initial_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[initial_idx].astype(float)

        # Initialize approximation sets
        L = [set() for _ in range(self.n_clusters)]  # Lower approximations
        U = [set() for _ in range(self.n_clusters)]  # Upper approximations

        iter_count = 0
        for iteration in range(self.max_iter):
            iter_count = iteration + 1
            old_centroids = centroids.copy()

            # Calculate cluster thresholds (alpha_j)
            alpha = np.zeros(self.n_clusters)
            for j in range(self.n_clusters):
                other_dists = [self._euclidean(centroids[j], centroids[m])
                               for m in range(self.n_clusters) if m != j]
                alpha[j] = 0.5 * min(other_dists) if other_dists else 0.0

            # Compute distance matrix between samples and centroids
            dist_matrix = np.array([
                [self._euclidean(x, mu) for mu in centroids]
                for x in X
            ])

            # Reset approximation sets for new iteration
            for j in range(self.n_clusters):
                L[j].clear()
                U[j].clear()

            # Assign samples to lower approximations (alpha threshold)
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    if dist_matrix[i, j] <= alpha[j]:
                        L[j].add(i)
                        U[j].add(i)

            # Calculate beta thresholds (maximum distance in upper approximations)
            beta = np.zeros(self.n_clusters)
            for j in range(self.n_clusters):
                beta[j] = max(dist_matrix[i, j] for i in U[j]) if U[j] else alpha[j]

            # Assign samples to upper approximations (beta threshold)
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    d = dist_matrix[i, j]
                    if alpha[j] < d <= beta[j]:
                        U[j].add(i)

            # Update centroids using weighted average of approximations
            new_centroids = np.zeros_like(centroids)
            for j in range(self.n_clusters):
                lower_idxs = list(L[j])
                fringe_idxs = [i for i in U[j] if i not in L[j]]

                if lower_idxs and fringe_idxs:
                    mu_L = X[lower_idxs].mean(axis=0)
                    mu_F = X[fringe_idxs].mean(axis=0)
                    new_centroids[j] = (
                        self.weight_lower * mu_L +
                        (1 - self.weight_lower) * mu_F
                    )
                elif lower_idxs:
                    new_centroids[j] = X[lower_idxs].mean(axis=0)
                elif fringe_idxs:
                    new_centroids[j] = X[fringe_idxs].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            centroids = new_centroids

            # Check convergence using centroid shifts
            shift = np.linalg.norm(centroids - old_centroids)
            if shift < self.tol:
                break

        # Convert approximation sets to binary matrices
        lower_matrix = np.zeros((n_samples, self.n_clusters), dtype=int)
        upper_matrix = np.zeros((n_samples, self.n_clusters), dtype=int)
        for j in range(self.n_clusters):
            for i in L[j]:
                lower_matrix[i, j] = 1
            for i in U[j]:
                upper_matrix[i, j] = 1

        return {
            'lower_approx': lower_matrix,
            'upper_approx': upper_matrix,
            'centroids': centroids,
            'n_iter': iter_count
        }
