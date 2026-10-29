# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for PossibilisticCMeans (PCM)."""

import numpy as np
import pytest
import scipy.sparse as sp

from soft_clustering import PCM
from soft_clustering._pcm import (
    _estimate_etas,
    _update_centers_from_T,
    _update_T_from_centers,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return np.vstack(
        [rng.normal([0, 0], 0.3, (30, 2)), rng.normal([5, 5], 0.3, (30, 2))]
    )


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


def test_weighted_spread_retains_small_positive_values_and_empty_support():
    X = np.array([[0.0], [2e-8]])
    weights = np.array([[1.0, 0.0, 1e-12], [1.0, 0.0, 1e-12]])
    with np.errstate(divide="raise", invalid="raise"):
        scales = _estimate_etas(X, weights, np.zeros((3, 1)), m=2, alpha=3)
    np.testing.assert_allclose(scales, [6e-16, 0.0, 6e-16], rtol=1e-14, atol=0)


def test_typicality_zero_limits_and_positive_subnormal_scales():
    step = 1e-160
    squared = step**2
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        actual = _update_T_from_centers(
            np.array([[0.0], [step]]),
            np.zeros((2, 1)),
            np.array([squared / 2, 0.0]),
            m=2,
        )
        unit = _update_T_from_centers(
            np.array([[0.0], [1.0]]), np.zeros((2, 1)), np.array([0.5, 0.0]), m=2
        )
    np.testing.assert_allclose(actual, [[1.0, 1.0], [1 / 3, 1e-12]], rtol=2e-13, atol=0)
    np.testing.assert_allclose(actual, unit, rtol=2e-13, atol=0)
    with pytest.raises(ValueError, match="nonnegative"):
        _update_T_from_centers(
            np.array([[0.0]]), np.array([[0.0]]), np.array([-1.0]), m=2
        )


def test_near_one_fuzzifier_typicalities_stay_finite():
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        actual = _update_T_from_centers(
            np.array([[0.0], [0.5], [2.0]]),
            np.array([[0.0]]),
            np.array([1.0]),
            m=1.000001,
        )
    np.testing.assert_array_equal(actual[:, 0], [1.0, 1.0, 1e-12])


def test_tiny_positive_mass_pcm_mean_and_empty_fallback():
    with np.errstate(divide="raise", invalid="raise"):
        centers = _update_centers_from_T(
            np.array([[1.0], [5.0]]),
            np.array([[0.0, 1e-12, 0.0], [1e-12, 0.0, 0.0]]),
            m=2,
        )
    np.testing.assert_array_equal(centers, [[5.0], [1.0], [0.0]])


@pytest.mark.parametrize("init", ["random", "kmeans++"])
def test_direct_initializers_use_only_data_derived_zero_spread_fallback(init):
    X = np.array([[1.0], [5.0]]) * 1e-7
    model = PCM(n_clusters=2, init=init, random_state=0, max_iter=5).fit(X)
    # Each seed has zero within-cluster spread; global ordered distances
    # are [0,16,16,0] times 1e-14, whose mean is 8e-14.
    np.testing.assert_allclose(model.etas_, 8e-14, rtol=1e-14, atol=0)
    assert np.isfinite(model.typicalities_).all()
    np.testing.assert_allclose(model.predict_proba(X), model.typicalities_)
