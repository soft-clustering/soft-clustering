# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for FuzzyCMeans (FCM)."""

import numpy as np
import pytest
import scipy.sparse as sp

from soft_clustering import FCM
from soft_clustering._fcm import _euclidean_dist2, _update_centers_from_U


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return np.vstack(
        [rng.normal([0, 0], 0.3, (30, 2)), rng.normal([5, 5], 0.3, (30, 2))]
    )


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


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_squared_distances_preserve_representable_differences(offset):
    X = np.array([[0.0], [1.0], [10.0], [11.0]]) + offset
    actual = _euclidean_dist2(X, X[[0, 2]])
    np.testing.assert_array_equal(
        actual, [[0.0, 100.0], [1.0, 81.0], [100.0, 0.0], [121.0, 1.0]]
    )


@pytest.mark.parametrize("dtype,offset", [(np.float32, 1e4), (np.float64, 1e9)])
def test_multidimensional_distances_duplicates_and_input_preservation(dtype, offset):
    X = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=dtype) + offset
    centers = X[[0, 1, 0]].copy()
    saved = (X.copy(), centers.copy())
    result = _euclidean_dist2(X, centers)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, [[0.0, 25.0, 0.0], [25.0, 0.0, 25.0]])
    np.testing.assert_array_equal(X, saved[0])
    np.testing.assert_array_equal(centers, saved[1])


def test_small_positive_distance_is_not_floored():
    actual = _euclidean_dist2(np.array([[1e-10]]), np.array([[0.0]]))
    np.testing.assert_allclose(actual, [[1e-20]], rtol=1e-14, atol=0)


@pytest.mark.parametrize("offset,mass", [(1e15, 1.0), (0.0, 1e-12)])
def test_weighted_mean_uses_positive_mass_and_keeps_empty_fallback(offset, mass):
    X = np.array([[offset], [offset + 1.0]])
    U = np.array([[mass, 0.0], [mass, 0.0]])
    with np.errstate(divide="raise", invalid="raise"):
        centers = _update_centers_from_U(X, U, m=2)
    np.testing.assert_array_equal(centers, [[offset + 0.5], [0.0]])


@pytest.mark.parametrize("offset", [0.0, 1e9, 1e15])
def test_translated_four_point_fit_keeps_groups_and_prediction(offset):
    X = np.array([[0.0], [1.0], [10.0], [11.0]]) + offset
    model = FCM(n_clusters=2, max_iter=50, random_state=0).fit(X)
    labels = model.labels_
    assert labels[0] == labels[1] != labels[2] == labels[3]
    for value in (model.memberships_, model.centers_, model.objective_trajectory_):
        assert np.isfinite(value).all()
    np.testing.assert_allclose(model.memberships_.sum(1), 1)
    np.testing.assert_allclose(model.predict_proba(X), model.memberships_, atol=1e-12)
    np.testing.assert_array_equal(model.predict(X), labels)
