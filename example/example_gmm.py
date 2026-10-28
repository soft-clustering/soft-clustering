# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""GMM: Gaussian mixture fitted by EM; memberships are posterior probabilities.

Run with ``python example/example_gmm.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import GMM

rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0.0, 0.0], 0.5, (40, 2)),
        rng.normal([4.0, 0.0], 0.5, (40, 2)),
        rng.normal([2.0, 3.5], 0.5, (40, 2)),
    ]
)

model = GMM(n_clusters=3, random_state=0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("mixing weights:", np.round(model.weights_, 3))
print(
    "log-likelihood never decreased:",
    bool(np.all(np.diff(model.log_likelihood_trajectory_) >= -1e-9)),
)
