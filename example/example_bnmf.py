# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""BayesianNMF: Bayesian NMF: relevance determination prunes unneeded communities.

Run with ``python example/example_bnmf.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import BayesianNMF

# A graph with three planted communities of 20 nodes each.
rng = np.random.default_rng(0)
communities = np.repeat([0, 1, 2], 20)
p = np.where(communities[:, None] == communities[None, :], 0.5, 0.03)
A = np.triu(rng.random((60, 60)) < p, 1).astype(float)
A = A + A.T

model = BayesianNMF(n_clusters=6, max_iter=500, random_state=0).fit(A)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("communities kept by relevance determination:", model.n_effective_, "of 6")
