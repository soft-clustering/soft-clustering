# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""SFCMEP: Semi-supervised fuzzy clustering from a few labels.

Run with ``python example/example_sfcmep.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import SFCMEP

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)
y = np.full(120, -1)  # -1 marks an unlabelled sample
y[[0, 1, 40, 41, 80, 81]] = [0, 0, 1, 1, 2, 2]

model = SFCMEP(n_clusters=3, random_state=0).fit(X, y)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
