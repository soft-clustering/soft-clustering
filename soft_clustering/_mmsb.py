import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.bernoulli import Bernoulli
from typeguard import typechecked


@typechecked
class MMSB:
    def __init__(self, n_nodes: int, n_blocks: int, alpha: float = 0.5):
        """
        Parameters:
        - n_nodes (int): Number of nodes in the graph
        - n_blocks (int): Number of latent blocks
        - alpha (float): Dirichlet prior parameter
        """
        self.n_nodes = n_nodes
        self.n_blocks = n_blocks
        self.alpha = alpha

        self.pi = Dirichlet(torch.full((n_blocks,), alpha)).sample((n_nodes,))
        # Random block interaction probabilities
        self.B = torch.rand((n_blocks, n_blocks))

    def sample_graph(self) -> torch.Tensor:
        """
        Generate a synthetic adjacency matrix based on the MMSB generative model.
        Returns:
        - Y (Tensor): Adjacency matrix (n_nodes x n_nodes)
        """
        Y = torch.zeros((self.n_nodes, self.n_nodes))

        for p in range(self.n_nodes):
            for q in range(self.n_nodes):
                z_p = torch.multinomial(self.pi[p], 1)
                z_q = torch.multinomial(self.pi[q], 1)
                prob = self.B[z_p, z_q]
                Y[p, q] = Bernoulli(prob).sample()

        return Y

    def get_memberships(self) -> torch.Tensor:
        """
        Returns:
        - pi (Tensor): Node membership distributions (n_nodes x n_blocks)
        """
        return self.pi

    def get_block_matrix(self) -> torch.Tensor:
        """
        Returns:
        - B (Tensor): Block interaction probability matrix (n_blocks x n_blocks)
        """
        return self.B
