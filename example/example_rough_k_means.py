# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""RoughKMeans: Rough k-means: lower and upper approximations.

Run with ``python example/example_rough_k_means.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import RoughKMeans

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)

model = RoughKMeans(n_clusters=3, threshold=1.3, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
boundary = model.lower_approx_.sum(axis=1) == 0
print("objects in a boundary region:", int(boundary.sum()))
