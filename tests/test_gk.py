# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for GustafsonKessel (GK)."""

import numpy as np
import pytest

from soft_clustering import GK, _gk


@pytest.fixture
def X():
    rng = np.random.default_rng(3)
    return np.vstack(
        [rng.normal([0, 0], 0.5, (30, 2)), rng.normal([5, 5], 0.5, (30, 2))]
    )


def test_fit_predict_runs(X):
    model = GK(random_state=0)
    result = model.fit_predict(X, K=2)
    assert result is not None


def test_memberships_shape(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    assert model.memberships_.shape == (60, 2)


def test_memberships_sum_to_one(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    np.testing.assert_allclose(model.memberships_.sum(axis=1), 1.0, atol=1e-5)


def test_centers_shape(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    assert model.centers_.shape == (2, 2)


def test_k3(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=3)
    assert model.memberships_.shape == (60, 3)


def test_covariances_stored(X):
    model = GK(random_state=0)
    model.fit_predict(X, K=2)
    assert hasattr(model, "covariances_")
    assert model.covariances_.shape == (2, 2, 2)


@pytest.mark.parametrize("offset", [0.0, 1e15])
@pytest.mark.parametrize("init", ["kmeans++", "random"])
def test_one_cluster_mean_is_not_attracted_to_zero(offset, init):
    X = np.array([[-1.0], [1.0]]) + offset
    model = GK(n_clusters=1, init=init, random_state=0).fit(X)
    np.testing.assert_array_equal(model.centers_, [[offset]])
    assert np.sum((X - model.centers_[0]) ** 2) == 2.0


def test_random_initialization_and_updates_use_true_positive_means(monkeypatch):
    X = np.array([[10.0], [11.0], [12.0], [13.0]])
    original = _gk._compute_covariances
    observed = []

    def covariance(data, U, centers, m, reg_covar):
        weights = U**m
        expected = np.array([np.average(data, axis=0, weights=w) for w in weights.T])
        np.testing.assert_allclose(centers, expected, rtol=2e-15, atol=0)
        assert np.all((centers >= 10.0) & (centers <= 13.0))
        observed.append(weights.sum(axis=0))
        return original(data, U, centers, m, reg_covar)

    monkeypatch.setattr(_gk, "_compute_covariances", covariance)
    GK(n_clusters=2, m=50.0, init="random", random_state=0, max_iter=2, tol=0.0).fit(X)
    assert len(observed) == 3  # Initial centers, updated centers, final covariance.
    assert min(np.min(mass) for mass in observed) < 1e-12


@pytest.mark.parametrize("mass_scale", [1.0, 1e-14, 0.0])
def test_positive_and_zero_weight_mean_and_scatter(mass_scale):
    X = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    U = np.column_stack([np.full(4, np.sqrt(mass_scale)), np.zeros(4)])
    centers = np.zeros((2, 2))
    ridge = 3e-6
    with np.errstate(divide="raise", invalid="raise"):
        means = _gk._update_centers_from_U(X + [10.0, 3.0], U, 2.0)
        cov = _gk._compute_covariances(X, U, centers, 2.0, reg_covar=ridge)
    expected = np.zeros((2, 2, 2))
    if mass_scale > 0:
        expected[0] = np.diag([2.0, 0.5])
    expected += ridge * np.eye(2)
    np.testing.assert_allclose(cov, expected, rtol=2e-15, atol=0)
    np.testing.assert_allclose(means[0], [10.0, 3.0] if mass_scale > 0 else [0.0, 0.0])
    np.testing.assert_array_equal(means[1], np.zeros(2))


def test_high_fuzzifier_centers_stay_in_positive_weight_convex_hull():
    X = np.array([[10.0], [11.0], [12.0], [13.0]])
    model = GK(n_clusters=2, m=50.0, random_state=0, max_iter=20, tol=0.0).fit(X)
    assert np.all((model.centers_ >= 10.0) & (model.centers_ <= 13.0))
