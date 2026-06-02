from os import path
import sys
import torch

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import MMSB

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
