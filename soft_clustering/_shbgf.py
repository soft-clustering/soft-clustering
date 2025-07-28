import numpy as np
from typing import List
from typeguard import typechecked


@typechecked
class SHBGF:
    def __init__(self, n_clusters: int, max_iter: int = 10) -> None:
        """
        Initializes the sHBGF algorithm.

        Parameters:
        - n_clusters (int): Number of final consensus clusters.
        - max_iter (int): Maximum number of iterations for KMeans.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit_predict(self, soft_memberships: List[np.ndarray]) -> np.ndarray:
        """
        Apply sHBGF consensus clustering on a list of soft membership matrices.

        Parameters:
        - soft_memberships (List[np.ndarray]): List of (N x Kq) matrices

        Returns:
        - labels (np.ndarray): Final cluster labels (shape N,)
        """
        from sklearn.cluster import KMeans

        N = soft_memberships[0].shape[0]
        concatenated = np.concatenate(soft_memberships, axis=1)

        # Normalize concatenated membership matrix (row-wise)
        concatenated /= concatenated.sum(axis=1, keepdims=True) + 1e-8

        # Run KMeans on concatenated soft assignments
        kmeans = KMeans(n_clusters=self.n_clusters,
                        max_iter=self.max_iter, n_init=10, random_state=42)
        labels = kmeans.fit_predict(concatenated)

        return labels
