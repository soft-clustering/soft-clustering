"""Unit and integration tests for PossibilisticCMeans (PCM)."""
import numpy as np
import pytest
import scipy.sparse as sp
from soft_clustering import PCM


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal([0, 0], 0.3, (30, 2)),
                      rng.normal([5, 5], 0.3, (30, 2))])


def test_output_shape(X):
    T = PCM(random_state=0).fit_predict(X, K=2)
    assert T.shape == (60, 2)


def test_typicalities_in_range(X):
    T = PCM(random_state=0).fit_predict(X, K=2)
    assert np.all(T >= 0) and np.all(T <= 1)


def test_centers_stored(X):
    model = PCM(random_state=0)
    model.fit_predict(X, K=2)
    assert model.centers_ is not None
    assert model.centers_.shape == (2, 2)


def test_etas_stored(X):
    model = PCM(random_state=0)
    model.fit_predict(X, K=2)
    assert model.etas_ is not None and len(model.etas_) == 2


def test_objective_trajectory(X):
    model = PCM(random_state=0)
    model.fit_predict(X, K=2)
    assert len(model.objective_trajectory_) >= 1


def test_random_init(X):
    T = PCM(random_state=2, init="random").fit_predict(X, K=2)
    assert T.shape == (60, 2)


def test_invalid_K_raises(X):
    with pytest.raises(ValueError):
        PCM().fit_predict(X, K=0)


def test_invalid_m_raises(X):
    with pytest.raises((ValueError, TypeError)):
        PCM(m=1.0).fit_predict(X, K=2)


def test_invalid_init_raises(X):
    with pytest.raises(ValueError):
        PCM(init="bogus").fit_predict(X, K=2)


def test_sparse_input(X):
    T = PCM(random_state=0).fit_predict(sp.csr_matrix(X), K=2)
    assert T.shape == (60, 2)


def test_k3_clusters(X):
    T = PCM(random_state=0).fit_predict(X, K=3)
    assert T.shape == (60, 3)
