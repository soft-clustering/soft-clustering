"""Unit and integration tests for SoftKSC (Soft Kernel Semi-supervised Clustering)."""

import numpy as np
import pytest
from soft_clustering import SoftKSC


@pytest.fixture
def data():
    rng = np.random.default_rng(33)
    X_lab = rng.normal(size=(10, 2))
    y_lab = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    X_unlab = rng.normal(size=(20, 2))
    return X_lab, y_lab, X_unlab


def test_fit_runs(data):
    X_lab, y_lab, X_unlab = data
    model = SoftKSC()
    model.fit(X_lab, y_lab, X_unlab)
    assert model.alpha is not None
    assert model.beta is not None


def test_predict_proba_shape(data):
    X_lab, y_lab, X_unlab = data
    model = SoftKSC()
    model.fit(X_lab, y_lab, X_unlab)
    probs = model.predict_proba(X_lab)
    assert probs.shape == (10, 2)


def test_predict_labels(data):
    X_lab, y_lab, X_unlab = data
    model = SoftKSC()
    model.fit(X_lab, y_lab, X_unlab)
    labels = model.predict(X_lab)
    assert labels.shape == (10,)
    assert set(labels).issubset({-1, 1})


def test_predict_proba_on_unlabeled(data):
    X_lab, y_lab, X_unlab = data
    model = SoftKSC()
    model.fit(X_lab, y_lab, X_unlab)
    probs = model.predict_proba(X_unlab)
    assert probs.shape == (20, 2)


def test_gamma_parameter(data):
    X_lab, y_lab, X_unlab = data
    model = SoftKSC(gamma=0.5)
    model.fit(X_lab, y_lab, X_unlab)
    assert model.predict(X_lab).shape == (10,)
