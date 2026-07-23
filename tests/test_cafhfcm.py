"""Unit and integration tests for CAFHFCM."""

import numpy as np
import pytest
from soft_clustering import CAFHFCM


@pytest.fixture
def X():
    rng = np.random.default_rng(10)
    return np.vstack(
        [rng.normal([0, 0], 0.4, (20, 2)), rng.normal([5, 5], 0.4, (20, 2))]
    )


def test_returns_labels_and_memberships(X):
    labels, U = CAFHFCM(c=2).fit_predict(X)
    assert labels.shape == (40,)
    assert U.shape == (40, 2)


def test_memberships_sum_to_one(X):
    _, U = CAFHFCM(c=2).fit_predict(X)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-5)


def test_memberships_nonneg(X):
    _, U = CAFHFCM(c=2).fit_predict(X)
    assert np.all(U >= 0)


def test_labels_in_range(X):
    labels, _ = CAFHFCM(c=2).fit_predict(X)
    assert set(labels).issubset({0, 1})


def test_centroids_stored(X):
    model = CAFHFCM(c=2)
    model.fit_predict(X)
    assert model.centroids is not None
    assert model.centroids.shape == (2, 2)


def test_k3(X):
    labels, U = CAFHFCM(c=3).fit_predict(X)
    assert U.shape == (40, 3)


def test_alpha_parameter(X):
    labels, U = CAFHFCM(c=2, alpha=0.5).fit_predict(X)
    assert U.shape == (40, 2)
