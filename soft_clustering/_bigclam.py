import numpy as np
from typeguard import typechecked


@typechecked
class BIGCLAM:
    def __init__(self, n_nodes: int, n_communities: int, max_iter: int = 100, learning_rate: float = 0.01):
        """
        Parameters:
        - n_nodes (int): Number of nodes in the graph
        - n_communities (int): Number of latent communities
        - max_iter (int): Maximum number of training iterations
        - learning_rate (float): Step size for gradient ascent
        """
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.F = np.random.rand(n_nodes, n_communities)  # Non-negative

    def fit(self, adj: np.ndarray):
        """
        Fit the BIGCLAM model to the adjacency matrix.

        Parameters:
        - adj (np.ndarray): Adjacency matrix (n x n), binary and symmetric
        """
        for _ in range(self.max_iter):
            grad = np.zeros_like(self.F)
            for u in range(self.n_nodes):
                neighbors = np.where(adj[u] > 0)[0]
                if len(neighbors) == 0:
                    continue
                f_u = self.F[u]
                for v in neighbors:
                    f_v = self.F[v]
                    inner = np.dot(f_u, f_v)
                    coeff = np.exp(-inner) / (1 - np.exp(-inner) + 1e-10)
                    grad[u] += f_v * coeff
                # negative gradient part
                sum_fv = np.sum(self.F, axis=0) - self.F[u]
                grad[u] -= sum_fv

            self.F += self.learning_rate * grad
            self.F = np.maximum(self.F, 1e-10)  # enforce non-negativity

    def get_membership(self) -> np.ndarray:
        """
        Returns:
        - F (np.ndarray): Membership matrix (n x k)
        """
        return self.F
