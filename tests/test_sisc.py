# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for SISC (Similarity-based Soft Clustering)."""

import numpy as np
import pytest

from soft_clustering import SISC


@pytest.fixture
def docs():
    return [
        "machine learning cluster algorithm",
        "fuzzy clustering membership centroid",
        "neural network deep learning",
        "graph community detection",
        "topic model document word",
        "probabilistic mixture model",
        "soft clustering assignment",
        "kernel distance metric",
    ]


def test_fit_predict_returns_dense_incidence(docs):
    """fit_predict returns memberships_, a dense binary incidence matrix.

    Earlier releases returned a scipy sparse matrix from fit_predict, unlike
    every other estimator.
    """
    model = SISC(n_clusters=2, random_state=0)
    result = model.fit_predict(docs)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, model.memberships_)
    assert set(np.unique(result)).issubset({0.0, 1.0})


def test_membership_rows(docs):
    model = SISC(random_state=0, k=3)
    result = model.fit_predict(docs)
    assert result.shape[0] == 8


def test_membership_columns(docs):
    # SISC merges candidate clusters, so the final count is discovered.
    model = SISC(random_state=0, k=3)
    result = model.fit_predict(docs)
    assert result.shape[1] >= 1
    assert model.n_clusters_ == result.shape[1]


def test_k2(docs):
    # Requesting fewer clusters must not yield more than requesting more.
    fewer = SISC(random_state=0, k=2).fit_predict(docs)
    more = SISC(random_state=0, k=4).fit_predict(docs)
    assert fewer.shape[1] <= more.shape[1]


def test_single_doc():
    model = SISC(random_state=0, k=2)
    result = model.fit_predict(["single document test"])
    assert result.shape[0] == 1
