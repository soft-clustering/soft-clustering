"""Unit and integration tests for RoughKMeans."""

import numpy as np
import pytest
from soft_clustering import RoughKMeans


@pytest.fixture
def X():
    rng = np.random.default_rng(18)
    return np.vstack(
        [rng.normal([0, 0], 0.4, (20, 2)), rng.normal([5, 5], 0.4, (20, 2))]
    )


def test_returns_dict(X):
    result = RoughKMeans(n_clusters=2, random_state=0).fit_predict(X)
    assert isinstance(result, dict)


def test_dict_keys(X):
    result = RoughKMeans(n_clusters=2, random_state=0).fit_predict(X)
    assert "lower_approx" in result
    assert "upper_approx" in result
    assert "centroids" in result
    assert "n_iter" in result


def test_lower_approx_shape(X):
    result = RoughKMeans(n_clusters=2, random_state=0).fit_predict(X)
    assert result["lower_approx"].shape == (40, 2)


def test_upper_approx_shape(X):
    result = RoughKMeans(n_clusters=2, random_state=0).fit_predict(X)
    assert result["upper_approx"].shape == (40, 2)


def test_upper_contains_lower(X):
    result = RoughKMeans(n_clusters=2, random_state=0).fit_predict(X)
    L = result["lower_approx"]
    U = result["upper_approx"]
    # Every lower-set member must be in the upper set
    assert np.all((L == 1) <= (U == 1))


def test_centroids_shape(X):
    result = RoughKMeans(n_clusters=2, random_state=0).fit_predict(X)
    assert result["centroids"].shape == (2, 2)


def test_n_iter_positive(X):
    result = RoughKMeans(n_clusters=2, random_state=0).fit_predict(X)
    assert result["n_iter"] >= 1


def test_too_few_samples_raises():
    X_small = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        RoughKMeans(n_clusters=3).fit_predict(X_small)


def test_k3(X):
    result = RoughKMeans(n_clusters=3, random_state=0).fit_predict(X)
    assert result["centroids"].shape == (3, 2)
