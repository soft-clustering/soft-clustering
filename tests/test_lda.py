# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for LDA (Latent Dirichlet Allocation)."""

import numpy as np
import pytest
import scipy.sparse as sp

from soft_clustering import LDA


@pytest.fixture
def docs():
    return [
        "machine learning data cluster",
        "fuzzy clustering algorithm membership",
        "neural network deep learning model",
        "graph nodes community detection",
        "topic model word document text",
        "probabilistic mixture latent variable",
        "soft assignment cluster center",
        "kernel similarity distance metric",
    ]


def test_fit_on_text(docs):
    model = LDA(n_topics=3, max_iter=5)
    model.fit(docs)
    assert hasattr(model, "gamma")


def test_gamma_shape(docs):
    model = LDA(n_topics=3, max_iter=5)
    model.fit(docs)
    assert model.gamma.shape == (8, 3)


def test_lambda_shape(docs):
    model = LDA(n_topics=3, max_iter=5)
    model.fit(docs)
    assert model.lambda_.shape[0] == 3  # K x V


def test_fit_on_matrix():
    rng = np.random.default_rng(34)
    X = rng.integers(0, 5, (10, 20))
    model = LDA(n_topics=3, max_iter=5)
    model.fit(X)
    assert model.gamma.shape == (10, 3)


def test_fit_on_sparse():
    rng = np.random.default_rng(35)
    X = sp.csr_matrix(rng.integers(0, 3, (10, 15)))
    model = LDA(n_topics=2, max_iter=5)
    model.fit(X)
    assert model.gamma.shape == (10, 2)


def test_vocab_attribute(docs):
    model = LDA(n_topics=2, max_iter=5)
    model.fit(docs)
    assert hasattr(model, "vocab_")


def test_k5_topics(docs):
    model = LDA(n_topics=5, max_iter=5)
    model.fit(docs)
    assert model.gamma.shape == (8, 5)


def test_duplicate_csr_counts_are_conserved_without_mutating_input():
    X = sp.csr_matrix(
        (np.array([2.0, 3.0, 1.0, 4.0]), np.array([0, 0, 1, 1]), np.array([0, 3, 4])),
        shape=(2, 2),
    )
    buffers = [X.data, X.indices, X.indptr]
    original = [a.tobytes() for a in buffers]
    for a in buffers:
        a.setflags(write=False)
    model = LDA(n_clusters=1, beta=0.01, max_iter=2, n_init=1, random_state=0).fit(X)
    np.testing.assert_allclose(model.lambda_, [[5.01, 5.01]], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.memberships_, np.ones((2, 1)))
    assert X.nnz == 4 and not X.has_canonical_format
    for actual, buffer, saved in zip((X.data, X.indices, X.indptr), buffers, original):
        assert actual is buffer
        assert actual.tobytes() == saved
        assert not actual.flags.writeable


@pytest.mark.parametrize("empty_document", [False, True])
def test_equivalent_count_representations_produce_the_same_fit(empty_document):
    if empty_document:
        dense = np.array([[0.0, 0.0], [2.0, 3.0]])
        duplicate = sp.csr_matrix(([1.0, 1.0, 3.0], [0, 0, 1], [0, 0, 3]), shape=(2, 2))
        explicit_zero = sp.csr_matrix(
            ([0.0, 2.0, 3.0], [0, 0, 1], [0, 1, 3]), shape=(2, 2)
        )
    else:
        dense = np.array([[5.0, 1.0], [0.0, 4.0]])
        duplicate = sp.csr_matrix(
            ([2.0, 3.0, 1.0, 4.0], [0, 0, 1, 1], [0, 3, 4]), shape=(2, 2)
        )
        explicit_zero = sp.csr_matrix(
            ([5.0, 1.0, 0.0, 4.0], [0, 1, 0, 1], [0, 2, 4]), shape=(2, 2)
        )
    kwargs = dict(
        n_clusters=2, alpha=1e-4, beta=0.01, max_iter=2, n_init=1, random_state=0
    )
    expected = LDA(**kwargs).fit(dense, vocabulary=["apple", "banana"])
    for X in (sp.csr_matrix(dense), duplicate, explicit_zero):
        original = [a.tobytes() for a in (X.data, X.indices, X.indptr)]
        model = LDA(**kwargs).fit(X, vocabulary=["apple", "banana"])
        for name in ("lambda_", "gamma", "memberships_"):
            actual = getattr(model, name)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(
                actual, getattr(expected, name), rtol=1e-12, atol=1e-12
            )
        np.testing.assert_array_equal(model.vocab_, ["apple", "banana"])
        assert [a.tobytes() for a in (X.data, X.indices, X.indptr)] == original
        if empty_document:
            np.testing.assert_array_equal(model.gamma[0], [1e-4, 1e-4])
            np.testing.assert_array_equal(model.memberships_[0], [0.5, 0.5])
