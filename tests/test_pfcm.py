"""Unit and integration tests for PFCM."""
import numpy as np
import pytest
from soft_clustering import PFCM


@pytest.fixture
def X():
    rng = np.random.default_rng(4)
    return np.vstack([rng.normal([0, 0], 0.5, (25, 2)),
                      rng.normal([6, 0], 0.5, (25, 2))])


def test_fit_returns_self(X):
    model = PFCM(n_clusters=2, random_state=0)
    ret = model.fit(X)
    assert ret is model


def test_membership_matrix_shape(X):
    model = PFCM(n_clusters=2, random_state=0).fit(X)
    assert model.membership_matrix.shape == (2, 50)


def test_typicality_matrix_shape(X):
    model = PFCM(n_clusters=2, random_state=0).fit(X)
    assert model.typicality_matrix.shape == (2, 50)


def test_membership_cols_sum_to_one(X):
    model = PFCM(n_clusters=2, random_state=0).fit(X)
    np.testing.assert_allclose(model.membership_matrix.sum(axis=0), 1.0, atol=1e-5)


def test_predict_memberships(X):
    model = PFCM(n_clusters=2, random_state=0).fit(X)
    M = model.predict_memberships(X)
    assert M.shape == (2, 50)


def test_predict_typicalities(X):
    model = PFCM(n_clusters=2, random_state=0).fit(X)
    T = model.predict_typicalities(X)
    assert T.shape == (2, 50)


def test_centroids_shape(X):
    model = PFCM(n_clusters=2, random_state=0).fit(X)
    assert model.cluster_centroids.shape == (2, 2)


def test_k3(X):
    model = PFCM(n_clusters=3, random_state=0).fit(X)
    assert model.membership_matrix.shape == (3, 50)
