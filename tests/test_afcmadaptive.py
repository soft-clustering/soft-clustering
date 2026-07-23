"""Unit and integration tests for AFCMAdaptive (image-based FCM)."""

import numpy as np
import pytest
from soft_clustering import AFCMAdaptive


@pytest.fixture
def image():
    rng = np.random.default_rng(13)
    return rng.random((20, 20)).astype(float)


def test_fit_runs(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    assert model.centers is not None


def test_predict_shape(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    labels = model.predict()
    assert labels.shape == (20, 20)


def test_labels_in_range(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    labels = model.predict()
    assert set(labels.flat).issubset({0, 1, 2})


def test_membership_shape(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    M = model.get_membership()
    assert M.shape == (20, 20, 3)


def test_membership_sums_to_one(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    M = model.get_membership()
    np.testing.assert_allclose(M.sum(axis=2), 1.0, atol=1e-5)


def test_k2(image):
    model = AFCMAdaptive(n_clusters=2, max_iter=5)
    model.fit(image)
    assert model.predict().shape == (20, 20)
