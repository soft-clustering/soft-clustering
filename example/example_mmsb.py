import sys
from os import path

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import MMSB

    n_nodes = 30
    n_blocks = 3

    # sample_graph() is the generative side of the model and needs no fit; it
    # is used here only to produce a graph to run inference on.
    Y = MMSB(
        n_blocks=n_blocks, alpha=0.5, n_nodes=n_nodes, random_state=0
    ).sample_graph()

    # get_memberships() and get_block_matrix() report the variational
    # posterior, so they require a fit against an observed adjacency matrix.
    model = MMSB(n_blocks=n_blocks, alpha=0.5, random_state=0).fit(Y)
    pi = model.get_memberships()
    B = model.get_block_matrix()

    print("Adjacency Matrix (Y):")
    print(Y)

    print("\nMembership Matrix (π):")
    print(pi)

    print("\nBlock Probability Matrix (B):")
    print(B)
