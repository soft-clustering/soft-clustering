# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""SoftKSC: Soft kernel spectral clustering, with out-of-sample assignment.

Run with ``python example/example_softksc.py`` after ``pip install soft-clustering``.
"""

import numpy as np
from sklearn.datasets import make_moons

from soft_clustering import SoftKSC

X, _ = make_moons(200, noise=0.05, random_state=0)
X_new, _ = make_moons(5, noise=0.05, random_state=1)

model = SoftKSC(n_clusters=2, gamma=10.0).fit(X)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("memberships of unseen points:\n", np.round(model.predict_proba(X_new), 3))
