"""Unit and integration tests for ECM (Evidential C-Means)."""
import numpy as np
import pytest
from soft_clustering import ECM


@pytest.fixture
def X():
    rng = np.random.default_rng(7)
    return np.vstack([rng.normal([0, 0], 0.5, (20, 2)),
                      rng.normal([5, 5], 0.5, (20, 2))])


def test_fit_runs(X):
    model = ECM(n_clusters=2)
    model.fit(X)
    assert model.mass is not None


def test_mass_shape(X):
    model = ECM(n_clusters=2)
    model.fit(X)
    # n_clusters + 1 (noise)
    assert model.get_membership().shape == (40, 3)


def test_mass_rows_sum_to_one(X):
    model = ECM(n_clusters=2)
    model.fit(X)
    M = model.get_membership()
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-6)


def test_mass_nonnegative(X):
    model = ECM(n_clusters=2)
    model.fit(X)
    assert np.all(model.get_membership() >= 0)


def test_prototypes_shape(X):
    model = ECM(n_clusters=2)
    model.fit(X)
    assert model.prototypes.shape == (2, 2)


def test_k3(X):
    model = ECM(n_clusters=3)
    model.fit(X)
    assert model.get_membership().shape == (40, 4)
