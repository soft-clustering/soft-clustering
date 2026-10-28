# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""SMCLA: Soft MCLA consensus of three FCM runs.

Run with ``python example/example_smcla.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import SMCLA

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)

from soft_clustering import FCM

# An ensemble of soft partitions: FCM with different seeds and fuzzifiers.
ensemble = [
    FCM(n_clusters=3, m=m, random_state=seed).fit_predict(X)
    for seed, m in ((0, 2.0), (1, 1.5), (2, 2.5))
]

model = SMCLA(n_clusters=3, random_state=0).fit(ensemble)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
