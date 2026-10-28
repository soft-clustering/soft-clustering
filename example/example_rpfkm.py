# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""RPFKM: Robust projected fuzzy k-means: clustering in a learned 1-D projection.

Run with ``python example/example_rpfkm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import RPFKM

rng = np.random.default_rng(0)
signal = np.concatenate([rng.normal(-4, 0.5, 40), rng.normal(4, 0.5, 40)])
X = np.column_stack([signal, rng.normal(0, 0.5, (80, 5))])  # one informative feature

model = RPFKM(n_clusters=2, d=1, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("projection weights:", np.round(model.projection_.ravel(), 2))
