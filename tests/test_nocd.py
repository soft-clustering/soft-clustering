"""Unit and integration tests for NOCD (Network Overlapping Community Detection)."""
import numpy as np
import pytest
import scipy.sparse as sp
from soft_clustering import NOCD


@pytest.fixture
def graph():
    rng = np.random.default_rng(37)
    n = 20
    A = (rng.random((n, n)) > 0.6).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    adj = sp.csr_matrix(A)
    features = sp.eye(n, format="csr")
    return adj, features, n


def test_fit_predict_runs(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16])
    memberships = model.fit_predict(adj, features, K=3)
    assert memberships is not None


def test_memberships_shape(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16])
    memberships = model.fit_predict(adj, features, K=3)
    assert memberships.shape == (n, 3)


def test_memberships_nonneg(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16])
    memberships = model.fit_predict(adj, features, K=3)
    assert np.all(memberships >= 0)


def test_k2(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16])
    memberships = model.fit_predict(adj, features, K=2)
    assert memberships.shape == (n, 2)
