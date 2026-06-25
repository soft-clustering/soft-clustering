"""Unit and integration tests for MBMM (Multivariate Beta Mixture Model)."""
import numpy as np
import pytest
from soft_clustering import MBMM


@pytest.fixture
def X():
    rng = np.random.default_rng(22)
    return rng.beta(2, 5, (50, 3))


def test_fit_runs(X):
    model = MBMM(n_components=3)
    model.fit(X)
    assert model.resp is not None


def test_predict_proba_shape(X):
    model = MBMM(n_components=3)
    model.fit(X)
    R = model.predict_proba()
    assert R.shape == (50, 3)


def test_responsibilities_sum_to_one(X):
    model = MBMM(n_components=3)
    model.fit(X)
    R = model.predict_proba()
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-6)


def test_predict_labels(X):
    model = MBMM(n_components=3)
    model.fit(X)
    labels = model.predict()
    assert labels.shape == (50,)
    assert set(labels).issubset({0, 1, 2})


def test_alpha_beta_shapes(X):
    model = MBMM(n_components=3)
    model.fit(X)
    assert model.alpha.shape == (3, 3)
    assert model.beta.shape == (3, 3)


def test_k2(X):
    model = MBMM(n_components=2)
    model.fit(X)
    assert model.predict_proba().shape == (50, 2)
