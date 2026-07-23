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
    model = WBSC(n_clusters=3)
    result = model.fit_predict(docs)
    assert result.shape[1] == 3


def test_k2(docs):
    model = WBSC(n_clusters=2)
    result = model.fit_predict(docs)
    assert result.shape[1] == 2


def test_single_doc():
    model = WBSC(n_clusters=2)
    result = model.fit_predict(["single document test"])
    assert result.shape[0] == 1
