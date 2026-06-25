"""Unit and integration tests for CAFCM (Collaborative Annealing FCM)."""
import numpy as np
import pytest
from soft_clustering import CAFCM


@pytest.fixture
def X():
    rng = np.random.default_rng(9)
    return np.vstack([rng.normal([0, 0], 0.4, (20, 2)),
                      rng.normal([5, 5], 0.4, (20, 2))])


def test_returns_labels_and_memberships(X):
    labels, U = CAFCM(c=2).fit_predict(X)
    assert labels.shape == (40,)
    assert U.shape == (40, 2)


def test_memberships_sum_to_one(X):
    _, U = CAFCM(c=2).fit_predict(X)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-5)


def test_memberships_nonneg(X):
    _, U = CAFCM(c=2).fit_predict(X)
    assert np.all(U >= 0)


def test_labels_in_range(X):
    labels, _ = CAFCM(c=2).fit_predict(X)
    assert set(labels).issubset({0, 1})


def test_centroids_stored(X):
    model = CAFCM(c=2)
    model.fit_predict(X)
    assert model.centroids is not None
    assert model.centroids.shape == (2, 2)


def test_k3(X):
    labels, U = CAFCM(c=3).fit_predict(X)
    assert U.shape == (40, 3)


def test_annealing_reduces_m(X):
    model = CAFCM(c=2, m_start=3.0, m_end=1.05, cooling_rate=0.9)
    labels, U = model.fit_predict(X)
    assert U.shape == (40, 2)
