# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""BGMM: Beta-Gaussian mixture over two aligned views.

Run with ``python example/example_bgmm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import BGMM

rng = np.random.default_rng(0)
gaussian_view = np.concatenate([rng.normal(-2, 0.5, 60), rng.normal(2, 0.5, 60)])
beta_view = np.clip(
    np.concatenate([rng.beta(2, 6, 60), rng.beta(6, 2, 60)]), 1e-3, 1 - 1e-3
)

model = BGMM(n_clusters=2, random_state=0).fit(gaussian_view, beta_view)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("Gaussian (mean, sd) per component:", np.round(model.gaussian_params, 2).tolist())
