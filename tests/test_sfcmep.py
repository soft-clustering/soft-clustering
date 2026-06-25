"""Unit and integration tests for SFCMEP (Semi-supervised FCM with Membership Prior)."""
import numpy as np
import pytest
from soft_clustering import SFCMEP


@pytest.fixture
def Xy():
    rng = np.random.default_rng(31)
    X = np.vstack([rng.normal([0, 0], 0.4, (20, 2)),
                   rng.normal([5, 5], 0.4, (20, 2))])
    # Label first 5 samples of each cluster; rest = None
    y = np.array([0]*5 + [None]*15 + [1]*5 + [None]*15, dtype=object)
    return X, y


def test_fit_predict_returns_dict(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert isinstance(result, dict)


def test_dict_keys(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert "centroids" in result
    assert "membership_matrix" in result


def test_membership_matrix_shape(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert result["membership_matrix"].shape == (40, 2)


def test_centroids_shape(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert result["centroids"].shape == (2, 2)


def test_membership_nonneg(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert np.all(result["membership_matrix"] >= 0)


def test_fully_unlabeled():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(20, 2))
    y = np.array([None] * 20, dtype=object)
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert result["membership_matrix"].shape == (20, 2)
