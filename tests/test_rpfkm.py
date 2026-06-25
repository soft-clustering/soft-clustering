"""Unit and integration tests for RPFKM (Robust Projection Fuzzy K-Means)."""
import numpy as np
import pytest
from soft_clustering import RPFKM


@pytest.fixture
def X():
    # RPFKM expects X of shape (D, N) — features x samples
    rng = np.random.default_rng(17)
    X_data = np.vstack([rng.normal([0, 0], 0.4, (25, 4)),
                        rng.normal([5, 5], 0.4, (25, 4))])
    return X_data.T  # (4, 50)


def test_returns_three_outputs(X):
    labels, U, W = RPFKM(c=2, d=2, random_state=0).fit_predict(X)
    assert labels is not None
    assert U is not None
    assert W is not None


def test_labels_shape(X):
    labels, U, W = RPFKM(c=2, d=2, random_state=0).fit_predict(X)
    assert labels.shape == (50,)


def test_labels_in_range(X):
    labels, _, _ = RPFKM(c=2, d=2, random_state=0).fit_predict(X)
    assert set(labels).issubset({0, 1})


def test_membership_shape(X):
    _, U, _ = RPFKM(c=2, d=2, random_state=0).fit_predict(X)
    assert U.shape == (2, 50)


def test_projection_shape(X):
    _, _, W = RPFKM(c=2, d=2, random_state=0).fit_predict(X)
    assert W.shape == (4, 2)


def test_k3(X):
    labels, U, W = RPFKM(c=3, d=2, random_state=0).fit_predict(X)
    assert U.shape == (3, 50)
