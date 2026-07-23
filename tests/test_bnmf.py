"""Unit and integration tests for BayesianNMF."""

import numpy as np
import pytest
from soft_clustering import BayesianNMF


@pytest.fixture
def V():
    rng = np.random.default_rng(23)
    return np.abs(rng.normal(size=(20, 20)))


def test_fit_runs(V):
    model = BayesianNMF(n_clusters=3, max_iter=20)
    model.fit(V)
    assert model.W is not None
    assert model.H is not None


def test_W_shape(V):
    model = BayesianNMF(n_clusters=3, max_iter=20)
    model.fit(V)
    assert model.W.shape == (20, 3)


def test_H_shape(V):
    model = BayesianNMF(n_clusters=3, max_iter=20)
    model.fit(V)
    assert model.H.shape == (3, 20)


def test_membership_nonneg(V):
    model = BayesianNMF(n_clusters=3, max_iter=20)
    model.fit(V)
    assert np.all(model.get_membership() >= 0)


def test_membership_shape(V):
    model = BayesianNMF(n_clusters=3, max_iter=20)
    model.fit(V)
    assert model.get_membership().shape == (20, 3)


def test_beta_stored(V):
    model = BayesianNMF(n_clusters=3, max_iter=20)
    model.fit(V)
    assert model.beta is not None
    assert len(model.beta) == 3


def test_k2(V):
    model = BayesianNMF(n_clusters=2, max_iter=20)
    model.fit(V)
    assert model.get_membership().shape == (20, 2)
