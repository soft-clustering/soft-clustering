import numpy as np
from typing import List
from typeguard import typechecked


@typechecked
class SMCLA:
    def __init__(self, n_clusters: int) -> None:
        """
        Initializes the sMCLA algorithm.

        Parameters:
        - n_clusters (int): Number of consensus clusters.
        """
        self.n_clusters = n_clusters

    def fit_predict(self, soft_memberships: List[np.ndarray]) -> np.ndarray:
        """
        Runs sMCLA on a list of soft membership matrices.

        Parameters:
        - soft_memberships (List[np.ndarray]): List of soft clusterings (N x Kq)

        Returns:
        - np.ndarray: Final consensus cluster labels
        """
        from sklearn.cluster import KMeans

        # Step 1: Stack all soft matrices (clusters) together
        cluster_vectors = []

        for soft in soft_memberships:
            cluster_vectors.extend(np.transpose(soft))

        cluster_vectors = np.array(cluster_vectors)  # shape: (sum(Kq), N)

        # Step 2: Cluster the meta-clusters
        kmeans_meta = KMeans(n_clusters=self.n_clusters,
                             random_state=42, n_init=10)
        meta_labels = kmeans_meta.fit_predict(cluster_vectors)

        # Step 3: Aggregate memberships per object using meta-cluster assignment
        N = soft_memberships[0].shape[0]
        consensus_matrix = np.zeros((N, self.n_clusters))

        for idx, vec in enumerate(cluster_vectors):
            cluster_id = meta_labels[idx]
            consensus_matrix[:, cluster_id] += vec

        # Step 4: Normalize each row and assign final label
        consensus_matrix /= consensus_matrix.sum(axis=1, keepdims=True) + 1e-8
        final_labels = np.argmax(consensus_matrix, axis=1)

        return final_labels
