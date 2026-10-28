# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""RDFKC: Robust deep fuzzy k-means with a small autoencoder (needs the deep extra).

Run with ``python example/example_rdfkc.py`` after ``pip install "soft-clustering[deep]"``.
"""

import numpy as np
import torch.nn as nn

from soft_clustering import RDFKC

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.4, (30, 8)) for c in (-2, 0, 2)]).astype("float32")
encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
decoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8))

model = RDFKC(
    n_clusters=3, encoder=encoder, decoder=decoder, random_state=0, max_iter=40, lr=1e-3
).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
