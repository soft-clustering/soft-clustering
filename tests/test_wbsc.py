# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for WBSC (Word-Based Soft Clustering)."""

import pytest
import scipy.sparse as sp

from soft_clustering import WBSC


@pytest.fixture
def docs():
    return [
        "machine learning cluster data algorithm",
        "fuzzy membership cluster centroid",
        "neural network deep learning training",
        "graph community detection nodes",
        "topic model word document",
        "probabilistic mixture gaussian",
        "soft clustering assignment",
        "kernel similarity distance",
    ]


def test_fit_predict_returns_sparse(docs):
    model = WBSC(n_clusters=3)
    result = model.fit_predict(docs)
    assert sp.issparse(result)


def test_membership_rows(docs):
    model = WBSC(n_clusters=3)
    result = model.fit_predict(docs)
    assert result.shape[0] == 8


def test_membership_columns(docs):
    # WBSC is non-parametric: it discovers the cluster count.
    model = WBSC()
    result = model.fit_predict(docs)
    assert result.shape[1] >= 1
    assert model.n_clusters == result.shape[1]


def test_k2(docs):
    # A stricter merge threshold must not increase the number of clusters.
    coarse = WBSC(similarity_threshold=0.1).fit_predict(docs)
    fine = WBSC(similarity_threshold=0.9).fit_predict(docs)
    assert coarse.shape[1] <= fine.shape[1]


def test_single_doc():
    model = WBSC(n_clusters=2)
    result = model.fit_predict(["single document test"])
    assert result.shape[0] == 1
