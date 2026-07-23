"""Unit and integration tests for BIGCLAM."""

import numpy as np
import pytest
from soft_clustering import BIGCLAM


@pytest.fixture
def adj():
    rng = np.random.default_rng(24)
    A = (rng.random((15, 15)) > 0.6).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def test_fit_runs(adj):
    model = BIGCLAM(n_nodes=15, n_communities=3, max_iter=10)
    model.fit(adj)
    assert model.F is not None


def test_membership_shape(adj):
    model = BIGCLAM(n_nodes=15, n_communities=3, max_iter=10)
    model.fit(adj)
    F = model.get_membership()
    assert F.shape == (15, 3)


def test_membership_nonneg(adj):
    model = BIGCLAM(n_nodes=15, n_communities=3, max_iter=10)
    model.fit(adj)
    assert np.all(model.get_membership() >= 0)


def test_k2(adj):
    model = BIGCLAM(n_nodes=15, n_communities=2, max_iter=10)
    model.fit(adj)
    assert model.get_membership().shape == (15, 2)


def test_isolated_node():
    adj = np.zeros((10, 10))
    model = BIGCLAM(n_nodes=10, n_communities=2, max_iter=5)
    model.fit(adj)  # all isolated — should not crash
    assert model.get_membership().shape == (10, 2)
