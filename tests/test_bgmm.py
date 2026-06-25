"""Unit and integration tests for BGMM (Bayesian Gaussian-Beta Mixture Model)."""
import numpy as np
import pytest
from soft_clustering import BGMM


@pytest.fixture
def data():
    rng = np.random.default_rng(21)
    Xg = rng.normal(0, 1, 60)
    Xb = rng.beta(2, 5, 60)
    return Xg, Xb


def test_fit_runs(data):
    Xg, Xb = data
    model = BGMM(n_components=3)
    model.fit(Xg, Xb)
    assert model.weights is not None


def test_predict_proba_shape(data):
    Xg, Xb = data
    model = BGMM(n_components=3).fit(Xg, Xb) or BGMM(n_components=3)
    model.fit(Xg, Xb)
    R = model.predict_proba()
    assert R.shape == (60, 3)


def test_responsibilities_sum_to_one(data):
    Xg, Xb = data
    model = BGMM(n_components=3)
    model.fit(Xg, Xb)
    R = model.predict_proba()
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-6)


def test_predict_labels_shape(data):
    Xg, Xb = data
    model = BGMM(n_components=3)
    model.fit(Xg, Xb)
    labels = model.predict()
    assert labels.shape == (60,)


def test_labels_in_range(data):
    Xg, Xb = data
    model = BGMM(n_components=3)
    model.fit(Xg, Xb)
    labels = model.predict()
    assert set(labels).issubset({0, 1, 2})


def test_weights_sum_to_one(data):
    Xg, Xb = data
    model = BGMM(n_components=3)
    model.fit(Xg, Xb)
    np.testing.assert_allclose(sum(model.weights), 1.0, atol=1e-5)


def test_gaussian_params_length(data):
    Xg, Xb = data
    model = BGMM(n_components=3)
    model.fit(Xg, Xb)
    assert len(model.gaussian_params) == 3


def test_n2_components(data):
    Xg, Xb = data
    model = BGMM(n_components=2)
    model.fit(Xg, Xb)
    assert model.predict_proba().shape == (60, 2)
