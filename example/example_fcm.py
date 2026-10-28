# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""FCM: Fuzzy c-means on three Gaussian blobs.

Run with ``python example/example_fcm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import FCM

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)

model = FCM(n_clusters=3, m=2.0, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("centers_:\n", np.round(model.centers_, 2))
print(
    "objective decreased:", bool(np.all(np.diff(model.objective_trajectory_) <= 1e-9))
)
print(
    "membership of a new point midway between two clusters:",
    np.round(model.predict_proba([[2.0, 0.0]]), 3),
)
