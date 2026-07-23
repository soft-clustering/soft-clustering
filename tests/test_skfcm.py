"""Unit and integration tests for SKFCM (Spatial Kernel FCM)."""

import numpy as np
import pytest
from soft_clustering import SKFCM


@pytest.fixture
def image_data():
    """Flatten a small 8x8 image to (64, 1) feature vector."""
    rng = np.random.default_rng(20)
    img = rng.random((8, 8))
    X = img.reshape(-1, 1)
    return X, (8, 8)


def test_fit_runs(image_data):
    X, shape = image_data
    model = SKFCM(n_clusters=3, max_iter=5)
    model.fit(X, shape)
    assert model.U is not None


def test_predict_shape(image_data):
    X, shape = image_data
    model = SKFCM(n_clusters=3, max_iter=5)
    model.fit(X, shape)
    labels = model.predict()
    assert labels.shape == (64,)


def test_predict_proba_shape(image_data):
    X, shape = image_data
    model = SKFCM(n_clusters=3, max_iter=5)
    model.fit(X, shape)
    U = model.predict_proba()
    assert U.shape == (64, 3)


def test_memberships_sum_to_one(image_data):
    X, shape = image_data
    model = SKFCM(n_clusters=3, max_iter=5)
    model.fit(X, shape)
    U = model.predict_proba()
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-4)


def test_labels_in_range(image_data):
    X, shape = image_data
    model = SKFCM(n_clusters=3, max_iter=5)
    model.fit(X, shape)
    labels = model.predict()
    assert set(labels).issubset({0, 1, 2})


def test_k2(image_data):
    X, shape = image_data
    model = SKFCM(n_clusters=2, max_iter=5)
    model.fit(X, shape)
    assert model.predict().shape == (64,)
