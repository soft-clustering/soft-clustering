"""Unit and integration tests for KFCM."""
import numpy as np
import pytest
from soft_clustering import KFCM


@pytest.fixture
def X():
    rng = np.random.default_rng(5)
    return np.vstack([rng.normal([0, 0], 0.5, (20, 2)),
                      rng.normal([4, 4], 0.5, (20, 2))])


def test_fit_returns_labels(X):
    labels = KFCM(n_clusters=2).fit(X)
    assert labels.shape == (40,)


def test_labels_valid_range(X):
    labels = KFCM(n_clusters=2).fit(X)
    assert set(labels).issubset({0, 1})


def test_membership_shape(X):
    model = KFCM(n_clusters=2)
    model.fit(X)
    assert model.U.shape == (2, 40)


def test_centers_shape(X):
    model = KFCM(n_clusters=2)
    model.fit(X)
    assert model.V.shape == (2, 2)


def test_invalid_n_clusters_raises():
    with pytest.raises(ValueError):
        KFCM(n_clusters=0)


def test_invalid_m_raises():
    with pytest.raises(ValueError):
        KFCM(m=1.0)


def test_invalid_sigma_raises():
    with pytest.raises(ValueError):
        KFCM(sigma=-1.0)


def test_empty_input_raises(X):
    model = KFCM(n_clusters=2)
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)))


def test_k3(X):
    labels = KFCM(n_clusters=3).fit(X)
    assert set(labels).issubset({0, 1, 2})
