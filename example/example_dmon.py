from os import path
import sys
import torch

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import DMoN

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long
    )
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
