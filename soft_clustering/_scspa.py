import numpy as np
from typing import List
from typeguard import typechecked


@typechecked
class SCSPA:
    def __init__(self, n_clusters: int) -> None:
        """
        Initializes the sCSPA algorithm.

        Parameters:
        - n_clusters (int): Number of consensus clusters.
        """
        self.n_clusters = n_clusters

    def fit_predict(self, soft_memberships: List[np.ndarray]) -> np.ndarray:
        """
        Runs sCSPA on a list of soft clustering matrices.

        Parameters:
        - soft_memberships (List[np.ndarray]): List of soft membership matrices (shape N x K)

        Returns:
        - np.ndarray: Final consensus cluster labels (shape N,)
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity

        # Step 1: Concatenate all soft membership matrices horizontally
        concatenated = np.concatenate(soft_memberships, axis=1)

        # Step 2: Normalize each row
        concatenated /= np.linalg.norm(concatenated,
                                       axis=1, keepdims=True) + 1e-8

        # Step 3: (Optional) Build cosine similarity matrix — not used directly
        _ = cosine_similarity(concatenated)

        # Step 4: Cluster with KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(concatenated)

        return labels
