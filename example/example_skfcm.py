# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""SKFCM: Spatially constrained kernel FCM on a noisy image.

Run with ``python example/example_skfcm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import SKFCM

rng = np.random.default_rng(0)
image = np.zeros((24, 24))
image[:, 12:] = 1.0
image += rng.normal(0, 0.3, image.shape)  # heavy noise

model = SKFCM(n_clusters=2, gamma=2.0, lambda_=0.5, random_state=0).fit(
    image.reshape(-1, 1), image.shape
)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
