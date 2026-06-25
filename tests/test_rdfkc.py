"""Unit and integration tests for RDFKC (Robust Deep Fuzzy K-Means Clustering)."""
import numpy as np
import pytest
import torch
from soft_clustering import RDFKC
from soft_clustering._rdfkc import FashionMNISTEncoder, FashionMNISTDecoder


@pytest.fixture
def X_fashion():
    """10 mini 28×28 grayscale images."""
    rng = np.random.default_rng(38)
    return rng.random((10, 1, 28, 28)).astype(np.float32)


@pytest.fixture
def fashion_model():
    return RDFKC(K=3, dataset="fashion", random_state=0, max_iter=2)


def test_fit_predict_runs(X_fashion, fashion_model):
    labels = fashion_model.fit_predict(X_fashion)
    assert labels is not None


def test_labels_shape(X_fashion, fashion_model):
    labels = fashion_model.fit_predict(X_fashion)
    assert labels.shape == (10,)


def test_labels_in_range(X_fashion, fashion_model):
    labels = fashion_model.fit_predict(X_fashion)
    assert set(labels.tolist()).issubset({0, 1, 2})


def test_tensor_input(X_fashion):
    model = RDFKC(K=2, dataset="fashion", random_state=0, max_iter=2)
    X_tensor = torch.from_numpy(X_fashion)
    labels = model.fit_predict(X_tensor)
    assert labels.shape == (10,)


def test_custom_encoder_decoder(X_fashion):
    enc = FashionMNISTEncoder()
    dec = FashionMNISTDecoder()
    model = RDFKC(K=2, encoder=enc, decoder=dec, random_state=0, max_iter=2)
    labels = model.fit_predict(X_fashion)
    assert labels.shape == (10,)


def test_invalid_dataset_raises():
    with pytest.raises(ValueError):
        RDFKC(K=2)
