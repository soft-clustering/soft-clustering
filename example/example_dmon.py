# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""DMoN: Deep modularity networks (needs the deep extra).

Run with ``python example/example_dmon.py`` after ``pip install "soft-clustering[deep]"``.
"""

import numpy as np

from soft_clustering import DMoN

# A graph with three planted communities of 20 nodes each.
rng = np.random.default_rng(0)
communities = np.repeat([0, 1, 2], 20)
p = np.where(communities[:, None] == communities[None, :], 0.5, 0.03)
A = np.triu(rng.random((60, 60)) < p, 1).astype(float)
A = A + A.T

model = DMoN(in_channels=60, hidden_channels=16, n_clusters=3, random_state=0)
model.fit(np.eye(60), adj=A)  # identity features

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
