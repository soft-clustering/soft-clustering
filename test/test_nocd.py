# Overview of NOCD usage
from os import path
import sys
import numpy as np
from scipy.sparse import csr_matrix


if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import NOCD


    n = 10
    p = 0.2

    upper = np.triu((np.random.rand(n, n) < p).astype(int), k=1)
    A = upper + upper.T
    adjacency_matrix = csr_matrix(A)

    feat = np.random.rand(n, n) * 0.1
    feat[:, :] += 1.0
    feature_matrix = csr_matrix(feat)

    K = 2  # number of communities

    # Initialize and fit the model
    model = NOCD(random_state=42, max_epochs=10)

    memberships = model.fit_predict(adjacency_matrix, feature_matrix, K)
    print("Membership matrix:\n", memberships)
