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
