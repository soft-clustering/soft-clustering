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
