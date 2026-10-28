# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""FCC: Fuzzy color clustering of CIELAB colors.

Run with ``python example/example_fcc.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import FCC

rng = np.random.default_rng(0)
X = np.vstack(
    [rng.normal(c, 3.0, (40, 3)) for c in ([30, 10, -20], [60, -30, 40], [85, 5, 5])]
)  # three color groups in L*a*b*

model = FCC(n_clusters=3, jnd=8.0, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
