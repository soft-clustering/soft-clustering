# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""AFCM: Adaptive fuzzy c-means with graph embedding on non-convex clusters.

Run with ``python example/example_afcm.py`` after ``pip install soft-clustering``.
"""

import numpy as np
from sklearn.datasets import make_moons

from soft_clustering import AFCM

X, y = make_moons(200, noise=0.05, random_state=0)

model = AFCM(n_clusters=2, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
from sklearn.metrics import adjusted_rand_score

print(
    "adjusted Rand index against the two moons:",
    round(adjusted_rand_score(y, model.labels_), 3),
)
