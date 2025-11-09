import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from soft_clustering._dmon._dmon import DMoN


def test_dmon():

    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                               [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long)
    x = torch.eye(6)  # identity features (6x6)
    adj = torch.zeros((6, 6))
    for i, j in edge_index.t():
        adj[i, j] = 1

    model = DMoN(in_channels=6, hidden_channels=8, n_clusters=2)
    soft_assign = model(x, edge_index, adj)
    loss = model.loss(soft_assign, adj)

    print("Soft assignments:")
    print(soft_assign)
    print("Loss:", loss.item())


if __name__ == "__main__":
    test_dmon()
