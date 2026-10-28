# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""SISC: Similarity-based soft clustering of documents.

Run with ``python example/example_sisc.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import SISC

docs = [
    "fuzzy clustering membership degree centroid",
    "fuzzy membership partition prototype degree",
    "clustering centroid prototype fuzzy partition",
    "graph community detection nodes edges",
    "network community nodes modularity graph",
    "edges nodes graph community network",
    "topic document word text corpus",
    "document text vocabulary language corpus",
    "word topic corpus document language",
]

model = SISC(n_clusters=3, random_state=0).fit(docs)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("unassigned documents (label -1):", int((model.labels_ == -1).sum()))
