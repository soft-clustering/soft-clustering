"""Unit and integration tests for SCM (Subtractive Clustering)."""
import numpy as np
import pytest
from soft_clustering import SCM


@pytest.fixture
def X():
    rng = np.random.default_rng(15)
    return np.vstack([rng.normal([0, 0], 0.3, (20, 2)),
                      rng.normal([5, 5], 0.3, (20, 2))])


def test_fit_returns_centers(X):
    model = SCM()
    centers = model.fit(X)
    assert centers is not None
    assert centers.ndim == 2
    assert centers.shape[1] == 2


def test_centers_stored(X):
    model = SCM()
    model.fit(X)
    assert model.centers_ is not None


def test_at_least_one_center(X):
    model = SCM()
    centers = model.fit(X)
    assert len(centers) >= 1


def test_centers_within_data_range(X):
    model = SCM()
    centers = model.fit(X)
    for center in centers:
        assert np.all(center >= X.min(axis=0) - 1)
        assert np.all(center <= X.max(axis=0) + 1)


def test_different_ra_values(X):
    for ra in [0.3, 0.5, 0.8]:
        model = SCM(ra=ra)
        centers = model.fit(X)
        assert centers.shape[1] == 2


def test_3d_input():
    rng = np.random.default_rng(16)
    X = rng.normal(size=(30, 3))
    model = SCM()
    centers = model.fit(X)
    assert centers.shape[1] == 3
