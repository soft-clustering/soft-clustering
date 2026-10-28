# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""FeMIFuzzy: Federated fuzzy clustering over two clients with missing values.

Run with ``python example/example_femifuzzy.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import FeMIFuzzy

rng = np.random.default_rng(0)
client_a = np.vstack([rng.normal(-3, 0.4, (15, 3)), rng.normal(3, 0.4, (15, 3))])
client_b = np.vstack([rng.normal(3, 0.4, (12, 3)), rng.normal(-3, 0.4, (12, 3))])
client_a[0, 1] = np.nan  # missing values are imputed locally
features = [["x", "y", "z"], ["x", "y", "z"]]

model = FeMIFuzzy(random_state=0).fit([client_a, client_b], features)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("cluster count chosen by each client:", model.client_n_clusters_)
