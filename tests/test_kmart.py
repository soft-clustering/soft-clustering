# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for KMART (Fuzzy ART document clustering)."""

import numpy as np
import pytest

from soft_clustering import KMART


@pytest.fixture
def docs():
    return [
        "machine learning data cluster algorithm",
        "fuzzy clustering membership centroid model",
        "neural network deep learning training",
        "graph community detection network nodes",
        "topic model latent word document text",
        "probabilistic mixture gaussian expectation",
        "soft assignment cluster membership degree",
        "kernel distance similarity metric function",
    ]


def test_fit_predict_returns_dense_incidence(docs):
    """fit_predict returns memberships_, a dense binary incidence matrix.

    Earlier releases returned a scipy sparse matrix from fit_predict, unlike
    every other estimator.
    """
    model = KMART(vigilance_param=0.3)
    result = model.fit_predict(docs)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, model.memberships_)
    assert set(np.unique(result)).issubset({0.0, 1.0})


def test_membership_rows(docs):
    model = KMART(vigilance_param=0.3)
    result = model.fit_predict(docs)
    assert result.shape[0] == 8


def test_clusters_formed(docs):
    model = KMART(vigilance_param=0.3)
    model.fit_predict(docs)
    assert len(model.clusters_) >= 1


def test_prototypes_formed(docs):
    model = KMART(vigilance_param=0.3)
    model.fit_predict(docs)
    assert len(model.prototypes_) >= 1


def test_cluster_words_formed(docs):
    model = KMART(vigilance_param=0.3)
    model.fit_predict(docs)
    assert len(model.cluster_words_) >= 1


def test_high_vigilance_more_clusters(docs):
    model_low = KMART(vigilance_param=0.1)
    model_low.fit_predict(docs)
    model_high = KMART(vigilance_param=0.9)
    model_high.fit_predict(docs)
    assert len(model_high.clusters_) >= len(model_low.clusters_)


def test_single_doc():
    model = KMART(vigilance_param=0.5)
    result = model.fit_predict(["single document test"])
    assert result.shape[0] == 1


def test_identical_documents_match_at_unit_vigilance():
    model = KMART(vigilance_param=1).fit(["apple", "apple", "apple"])
    np.testing.assert_array_equal(model.memberships_, np.ones((3, 1)))
    assert model.clusters_ == [{0, 1, 2}]
    np.testing.assert_array_equal(model.prototypes_, [[1.0]])


@pytest.mark.parametrize("rho", [np.nextafter(0.5, 0), 0.5, np.nextafter(0.5, 1)])
def test_vigilance_is_inclusive_at_the_exact_half_boundary(rho):
    model = KMART(vigilance_param=rho).fit(["apple", "apple banana"])
    expected = np.ones((2, 1)) if rho <= 0.5 else np.eye(2)
    np.testing.assert_array_equal(model.memberships_, expected)


@pytest.mark.parametrize("rho", [0.0, 0.5])
@pytest.mark.parametrize("docs", [["apple", "", "the and"], ["", "the and", "and"]])
def test_empty_documents_keep_zero_score_policy(rho, docs):
    model = KMART(vigilance_param=rho, learning_rate=0.4).fit(docs)
    expected = np.ones((3, 1)) if rho == 0 else np.eye(3)
    np.testing.assert_array_equal(model.memberships_, expected)
    assert model.clusters_ == ([{0, 1, 2}] if rho == 0 else [{0}, {1}, {2}])
    if docs[0]:
        expected_prototypes = [[0.36]] if rho == 0 else [[1.0], [0.0], [0.0]]
        np.testing.assert_allclose(model.prototypes_, expected_prototypes)
    else:
        assert all(p.size == 0 for p in model.prototypes_)
        assert all(words == set() for words in model.cluster_words_)


def test_boundary_matches_preserve_overlap_partial_updates_and_keywords():
    docs = ["apple apple", "banana banana", "apple banana", "apple banana"]
    model = KMART(learning_rate=0.4).fit(docs)
    np.testing.assert_array_equal(model.memberships_, [[1, 0], [0, 1], [1, 1], [1, 1]])
    assert model.clusters_ == [{0, 2, 3}, {1, 2, 3}]
    np.testing.assert_allclose(
        model.prototypes_, [[1.36, 0], [0, 1.36]], rtol=0, atol=1e-15
    )
    assert model.cluster_words_ == [{"apple", "banana"}, {"apple", "banana"}]
