"""Unit and integration tests for GaussianMixtureEM (GMM)."""

import numpy as np
import pytest
from soft_clustering import GMM


@pytest.fixture
def X():
    rng = np.random.default_rng(2)
    return np.vstack(
        [rng.normal([0, 0], 0.5, (40, 2)), rng.normal([6, 0], 0.5, (40, 2))]
    )


def test_fit_predict_returns_labels(X):
    model = GMM(random_state=0)
    result = model.fit_predict(X, K=2)
    assert result is not None


def test_responsibilities_shape(X):
    model = GMM(random_state=0)
    model.fit_predict(X, K=2)
    assert model.responsibilities_.shape == (80, 2)


def test_responsibilities_sum_to_one(X):
    model = GMM(random_state=0)
    model.fit_predict(X, K=2)
    np.testing.assert_allclose(model.responsibilities_.sum(axis=1), 1.0, atol=1e-6)


def test_means_shape(X):
    model = GMM(random_state=0)
    model.fit_predict(X, K=2)
    assert model.means_.shape == (2, 2)


def test_weights_sum_to_one(X):
    model = GMM(random_state=0)
    model.fit_predict(X, K=2)
    np.testing.assert_allclose(model.weights_.sum(), 1.0, atol=1e-6)


def test_covariance_types():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 2))
    for cov_type in ["full", "diag", "spherical"]:
        model = GMM(covariance_type=cov_type, random_state=0)
        model.fit_predict(X, K=2)
        assert model.responsibilities_.shape == (30, 2)


def test_k3(X):
    model = GMM(random_state=0)
    model.fit_predict(X, K=3)
    assert model.responsibilities_.shape == (80, 3)


def test_log_likelihoods_non_empty(X):
    model = GMM(random_state=0)
    model.fit_predict(X, K=2)
    assert hasattr(model, "log_likelihoods_")
    assert len(model.log_likelihoods_) >= 1
