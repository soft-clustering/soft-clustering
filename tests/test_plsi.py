# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for PLSI (Probabilistic Latent Semantic Indexing)."""

import numpy as np
import pytest
import scipy.sparse as sp

from soft_clustering import PLSI


@pytest.fixture
def docs():
    return [
        "machine learning data cluster algorithm",
        "fuzzy clustering membership centroid",
        "neural network deep learning training",
        "graph community detection network nodes",
        "topic model latent word document text",
        "probabilistic mixture gaussian expectation",
        "soft assignment cluster membership degree",
        "kernel distance similarity metric function",
    ]


def test_fit_predict_on_text_runs(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_w_given_z is not None


def test_P_w_given_z_shape(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_w_given_z.shape[0] == 3


def test_P_d_given_z_shape(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_d_given_z.shape == (3, 8)


def test_P_z_shape(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_z.shape == (3,)
    np.testing.assert_allclose(model.P_z.sum(), 1.0, atol=1e-5)


def test_fit_predict_on_sparse():
    rng = np.random.default_rng(36)
    X = sp.csr_matrix(rng.integers(0, 5, (10, 15)).astype(float))
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(X)
    assert model.P_w_given_z is not None


def test_k2(docs):
    model = PLSI(n_components=2, max_iter=5)
    model.fit_predict(docs)
    assert model.P_d_given_z.shape == (2, 8)


def test_vocab_set_on_text(docs):
    model = PLSI(n_components=2, max_iter=5)
    model.fit_predict(docs)
    assert model.vocab is not None


def test_square_posterior_has_canonical_orientation():
    model = PLSI(n_topics=2, max_iter=1, n_init=1, random_state=0)
    assert model.fit(["cats cats cats dogs", "dogs cars"]) is model
    assert model.memberships_.shape == (2, 2)
    assert np.isfinite(model.memberships_).all()
    np.testing.assert_allclose(model.memberships_.sum(axis=1), 1)
    np.testing.assert_array_equal(model.memberships_, model.P_z_given_d.T)
    assert model.labels_.shape == (2,)
    assert (
        model.n_clusters_ == model.n_clusters == model.get_params()["n_clusters"] == 2
    )


@pytest.mark.parametrize(
    "data",
    [
        ["cats dogs", "cars roads", "flowers trees"],
        sp.csr_matrix([[2, 1, 0], [0, 3, 1], [1, 0, 4]]),
    ],
)
def test_keyword_and_positional_canonical_and_legacy_arrays_agree(data):
    first = PLSI(n_clusters=2, max_iter=2, n_init=1, random_state=0)
    second = PLSI(n_clusters=2, max_iter=2, n_init=1, random_state=0)
    result = first.fit_predict(data)
    keyword = second.fit_predict(data=data)
    assert result is first.memberships_
    assert keyword is second.memberships_
    assert result.shape == (3, 2)
    for attr in (
        "memberships_",
        "labels_",
        "P_z_given_d",
        "P_d_given_z",
        "P_w_given_z",
    ):
        np.testing.assert_array_equal(getattr(first, attr), getattr(second, attr))
    assert first.P_z_given_d.shape == first.P_d_given_z.shape == (2, 3)
    assert first.P_w_given_z.shape[0] == 2
    words = first.get_P_w_given_d()
    assert words.shape == (3, first.P_w_given_z.shape[1])
    np.testing.assert_allclose(words, first.memberships_ @ first.P_w_given_z)
