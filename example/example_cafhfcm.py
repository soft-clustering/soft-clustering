# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""CAFHFCM: Centroid auto-fused hierarchical FCM: the number of clusters is discovered.

Run with ``python example/example_cafhfcm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import CAFHFCM

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)

model = CAFHFCM(alpha=0.01, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print(
    "(initial centroids, after fusion):",
    model.merge_history_[-1] if model.merge_history_ else None,
)
