"""Unit and integration tests for AFCM (Adaptive FCM with graph embedding)."""

import numpy as np
import pytest
from soft_clustering import AFCM


@pytest.fixture
def X():
    rng = np.random.default_rng(11)
    return np.vstack(
        [rng.normal([0, 0], 0.4, (20, 2)), rng.normal([5, 5], 0.4, (20, 2))]
    )


def test_returns_labels_and_memberships(X):
    labels, U = AFCM(c=2, n_neighbors=4).fit_predict(X)
    assert labels.shape == (40,)
    assert U.shape == (40, 2)


def test_labels_in_range(X):
    labels, _ = AFCM(c=2, n_neighbors=4).fit_predict(X)
    assert set(labels).issubset({0, 1})


def test_memberships_nonneg(X):
    _, U = AFCM(c=2, n_neighbors=4).fit_predict(X)
    assert np.all(U >= 0)


def test_memberships_sum_to_one(X):
    _, U = AFCM(c=2, n_neighbors=4).fit_predict(X)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-5)


def test_k3(X):
    labels, U = AFCM(c=3, n_neighbors=4).fit_predict(X)
    assert U.shape == (40, 3)


def test_lambda_parameter(X):
    labels, U = AFCM(c=2, lambda_=2.0, n_neighbors=4).fit_predict(X)
    assert U.shape == (40, 2)
