# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""PFCM: Possibilistic fuzzy c-means: memberships and typicalities.

Run with ``python example/example_pfcm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import PFCM

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)
X = np.vstack([X, [[20.0, 20.0]]])  # one far outlier

model = PFCM(n_clusters=3, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("outlier membership (sums to one):", np.round(model.memberships_[-1], 3))
print("outlier typicality (all small):  ", np.round(model.typicalities_[-1], 4))
