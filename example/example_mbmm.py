# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""MBMM: Mixture of Beta distributions for data in (0, 1).

Run with ``python example/example_mbmm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import MBMM

rng = np.random.default_rng(0)
X = np.vstack([rng.beta(2, 8, (60, 3)), rng.beta(8, 2, (60, 3))])
X = np.clip(X, 1e-3, 1 - 1e-3)

model = MBMM(n_clusters=2, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
