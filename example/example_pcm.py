# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""PCM: Possibilistic c-means: typicalities need not sum to one.

Run with ``python example/example_pcm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import PCM

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)
X = np.vstack([X, [[20.0, 20.0]]])  # one far outlier

model = PCM(n_clusters=3, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("typicalities of the outlier (all small):", np.round(model.memberships_[-1], 4))
