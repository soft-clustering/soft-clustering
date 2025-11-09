import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked


@typechecked
class CDCGS(nn.Module):
    def __init__(self, num_nodes: int, n_clusters: int, tau: float = 1.0):
        super(CDCGS, self).__init__()
        self.num_nodes = num_nodes
        self.n_clusters = n_clusters
        self.tau = tau  # Gumbel-Softmax temperature

        self.W_C = nn.Parameter(torch.randn(num_nodes, n_clusters))

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        # W_C -> soft cluster assignment using Gumbel-Softmax
        soft_assign = F.gumbel_softmax(
            self.W_C, tau=self.tau, hard=False, dim=1)  # (n x k)

        # Compute R = W_C^T A W_C
        R = soft_assign.T @ adj @ soft_assign  # (k x k)
        output = F.softmax(R, dim=1)  # normalize rows

        return output, soft_assign

    def loss(self, output: torch.Tensor) -> torch.Tensor:
        # Simple structure loss: encourage diagonal R
        identity = torch.eye(self.n_clusters, device=output.device)
        return F.mse_loss(output, identity)
