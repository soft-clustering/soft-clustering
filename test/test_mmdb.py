import torch
from soft_clustering._mmsb._mmsb import MMSB


def test_mmsb():
    n_nodes = 6
    n_blocks = 3
    model = MMSB(n_nodes=n_nodes, n_blocks=n_blocks, alpha=0.5)

    Y = model.sample_graph()
    pi = model.get_memberships()
    B = model.get_block_matrix()

    print("Adjacency Matrix (Y):")
    print(Y)

    print("\nMembership Matrix (π):")
    print(pi)

    print("\nBlock Probability Matrix (B):")
    print(B)


if __name__ == "__main__":
    test_mmsb()
