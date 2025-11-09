import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch_geometric.nn import GCNConv


@typechecked
class DMoN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, n_clusters: int):
        super(DMoN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, n_clusters)
        self.n_clusters = n_clusters

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        soft_assign = F.softmax(self.conv2(h, edge_index), dim=1)
        return soft_assign

    def loss(self, soft_assign: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        m = adj.sum() / 2
        degrees = adj.sum(dim=1, keepdim=True)
        B = adj - (degrees @ degrees.T) / (2 * m + 1e-8)
        modularity_term = torch.trace(
            soft_assign.T @ B @ soft_assign) / (2 * m + 1e-8)

        cluster_size = soft_assign.sum(dim=0)
        collapse_reg = (torch.norm(cluster_size, p=2) /
                        soft_assign.shape[0] - 1) ** 2

        return -modularity_term + collapse_reg
