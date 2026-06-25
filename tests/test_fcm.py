"""Unit and integration tests for FuzzyCMeans (FCM)."""
import numpy as np
import pytest
import scipy.sparse as sp
from soft_clustering import FCM


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal([0, 0], 0.3, (30, 2)),
                      rng.normal([5, 5], 0.3, (30, 2))])


def test_output_shape(X):
    model = FCM(random_state=0)
    U = model.fit_predict(X, K=2)
    assert U.shape == (60, 2)


def test_memberships_sum_to_one(X):
    U = FCM(random_state=0).fit_predict(X, K=2)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-6)


def test_memberships_nonnegative(X):
    U = FCM(random_state=0).fit_predict(X, K=2)
    assert np.all(U >= 0)


def test_centers_stored(X):
    model = FCM(random_state=0)
    model.fit_predict(X, K=2)
    assert model.centers_ is not None
    assert model.centers_.shape == (2, 2)


def test_objective_trajectory(X):
    model = FCM(random_state=0)
    model.fit_predict(X, K=2)
    assert model.objective_trajectory_ is not None
    assert len(model.objective_trajectory_) >= 1


def test_random_init(X):
    model = FCM(random_state=1, init="random")
    U = model.fit_predict(X, K=2)
    assert U.shape == (60, 2)


def test_k_clusters_3(X):
    U = FCM(random_state=0).fit_predict(X, K=3)
    assert U.shape == (60, 3)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-6)


def test_sparse_input(X):
    X_sparse = sp.csr_matrix(X)
    U = FCM(random_state=0).fit_predict(X_sparse, K=2)
    assert U.shape == (60, 2)


def test_invalid_K_raises(X):
    with pytest.raises(ValueError):
        FCM().fit_predict(X, K=0)


def test_invalid_m_raises(X):
    with pytest.raises((ValueError, TypeError)):
        FCM(m=1.0).fit_predict(X, K=2)


def test_invalid_init_raises(X):
    with pytest.raises(ValueError):
        FCM(init="bad").fit_predict(X, K=2)


def test_convergence_separable_data(X):
    model = FCM(random_state=0, max_iter=500, tol=1e-8)
    U = model.fit_predict(X, K=2)
    labels = np.argmax(U, axis=1)
    # Expect good separation (at most a handful of misclassifications)
    purity = max(
        np.mean(labels[:30] == 0) + np.mean(labels[30:] == 1),
        np.mean(labels[:30] == 1) + np.mean(labels[30:] == 0),
    )
    assert purity >= 1.5  # sum of two accuracies, so >= 1.5 means good


def test_reproducibility(X):
    U1 = FCM(random_state=42).fit_predict(X, K=2)
    U2 = FCM(random_state=42).fit_predict(X, K=2)
    np.testing.assert_array_equal(U1, U2)
