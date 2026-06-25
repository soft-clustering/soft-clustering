"""Unit and integration tests for GustafsonKessel (GK)."""
import numpy as np
import pytest
from soft_clustering import GK


@pytest.fixture
def X():
    rng = np.random.default_rng(3)
    return np.vstack([rng.normal([0, 0], 0.5, (30, 2)),
                      rng.normal([5, 5], 0.5, (30, 2))])


def test_fit_predict_runs(X):
    model = GK(random_state=0)
    result = model.fit_predict(X, K=2)
    assert result is not None


def test_memberships_shape(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    assert model.memberships_.shape == (60, 2)


def test_memberships_sum_to_one(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    np.testing.assert_allclose(model.memberships_.sum(axis=1), 1.0, atol=1e-5)


def test_centers_shape(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    assert model.centers_.shape == (2, 2)


def test_k3(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=3)
    assert model.memberships_.shape == (60, 3)


def test_covariances_stored(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    assert hasattr(model, "covariances_")
    assert model.covariances_.shape == (2, 2, 2)
