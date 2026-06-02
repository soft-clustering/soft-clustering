from os import path
import sys
import torch

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import CDCGS

    adj = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0]
    ], dtype=torch.float)

    model = CDCGS(num_nodes=6, n_clusters=2, tau=1.0)
    R, soft_assign = model(adj)
    loss = model.loss(R)

    print("Soft Cluster Assignment:")
    print(soft_assign)

    print("Cluster Relation Matrix (R):")
    print(R)

    print("Loss:", loss.item())
