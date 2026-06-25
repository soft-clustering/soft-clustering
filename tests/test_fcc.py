"""Unit and integration tests for FCC (Fuzzy Color Clustering)."""
import numpy as np
import pytest
from soft_clustering import FCC


@pytest.fixture
def X():
    rng = np.random.default_rng(14)
    # CIELAB-like data in 3D
    return rng.uniform(0, 100, (40, 3))


def test_returns_labels_and_memberships(X):
    labels, U = FCC(c=3).fit_predict(X)
    assert labels.shape == (40,)
    assert U.shape == (40, 3)


def test_memberships_nonneg(X):
    _, U = FCC(c=3).fit_predict(X)
    assert np.all(U >= 0)


def test_memberships_sum_to_one(X):
    _, U = FCC(c=3).fit_predict(X)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-5)


def test_labels_in_range(X):
    labels, _ = FCC(c=3).fit_predict(X)
    assert set(labels).issubset({0, 1, 2})


def test_centroids_shape(X):
    model = FCC(c=3)
    model.fit_predict(X)
    assert model.centroids.shape == (3, 3)


def test_jnd_parameter(X):
    labels, U = FCC(c=3, jnd=50.0).fit_predict(X)
    assert U.shape == (40, 3)


def test_k2(X):
    labels, U = FCC(c=2).fit_predict(X)
    assert U.shape == (40, 2)
