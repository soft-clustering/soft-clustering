"""Unit and integration tests for KMART (Fuzzy ART document clustering)."""

import numpy as np
import pytest
import scipy.sparse as sp
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


def test_fit_predict_returns_sparse(docs):
    model = KMART(vigilance_param=0.3)
    result = model.fit_predict(docs)
    assert sp.issparse(result)


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
