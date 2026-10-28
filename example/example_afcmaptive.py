# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""AFCMAdaptive: Adaptive FCM segmentation of an image with a smooth intensity bias.

Run with ``python example/example_afcmaptive.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import AFCMAdaptive

rng = np.random.default_rng(0)
truth = np.zeros((32, 32), dtype=int)
truth[:, 11:22] = 1
truth[:, 22:] = 2
yy, xx = np.mgrid[0:32, 0:32] / 32.0
bias = 1.0 + 0.4 * (xx - 0.5)  # multiplicative inhomogeneity
image = np.array([0.2, 0.5, 0.8])[truth] * bias + rng.normal(0, 0.02, (32, 32))

model = AFCMAdaptive(n_clusters=3).fit(image)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("segmentation (first row):", model.label_map()[0])
print(
    "estimated gain field range:",
    np.round([model.gain_field_.min(), model.gain_field_.max()], 3),
)
