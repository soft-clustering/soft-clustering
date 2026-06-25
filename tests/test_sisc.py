"""Unit and integration tests for SISC (Similarity-based Soft Clustering)."""
import pytest
import scipy.sparse as sp
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


def test_fit_predict_returns_sparse(docs):
    model = SISC(k=3)
    result = model.fit_predict(docs)
    assert sp.issparse(result)


def test_membership_rows(docs):
    model = SISC(k=3)
    result = model.fit_predict(docs)
    assert result.shape[0] == 8


def test_membership_columns(docs):
    model = SISC(k=3)
    result = model.fit_predict(docs)
    assert result.shape[1] == 3


def test_k2(docs):
    model = SISC(k=2)
    result = model.fit_predict(docs)
    assert result.shape[1] == 2


def test_single_doc():
    model = SISC(k=2)
    result = model.fit_predict(["single document test"])
    assert result.shape[0] == 1
