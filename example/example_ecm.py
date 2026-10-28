# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""ECM: Evidential c-means: a credal partition with ambiguity and outliers.

Run with ``python example/example_ecm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import ECM

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)
X = np.vstack([X, [[2.0, 0.0], [30.0, 30.0]]])  # ambiguous point, outlier

model = ECM(n_clusters=3, delta=5.0, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
sets = [
    "{" + ",".join(str(k) for k in np.flatnonzero(f)) + "}" for f in model.focal_sets_
]
for name, row in (("ambiguous point", -2), ("outlier", -1)):
    masses = dict(zip(["{}"] + sets, np.round(model.masses_[row], 3).tolist()))
    print(name, "masses:", masses)
