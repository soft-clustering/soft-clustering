# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""KMART: Modified fuzzy ART for documents.

Run with ``python example/example_kmart.py`` after ``pip install soft-clustering``.
"""

import numpy as np

from soft_clustering import KMART

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

model = KMART(vigilance_param=0.3).fit(docs)

print("memberships_ shape:", model.memberships_.shape)
print("first three rows:\n", np.round(model.memberships_[:3], 3))
print("labels_:", model.labels_)
print("clusters produced:", model.n_clusters_)
print("keywords per category:", [sorted(w)[:4] for w in model.cluster_words_])
