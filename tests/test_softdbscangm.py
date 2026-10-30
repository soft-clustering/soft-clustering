# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for SoftDBSCANGM."""

import numpy as np
import pytest

from soft_clustering import SoftDBSCANGM
from soft_clustering.benchmarking import soft_clustering_metrics


@pytest.mark.parametrize("offset", [0.0, 1e12])
def test_one_cluster_uses_arithmetic_mean(offset):
    X = np.array([[-1.0], [1.0]]) + offset
    model = SoftDBSCANGM(eps=3.0, min_samples=1, max_iter=1).fit(X)
    np.testing.assert_array_equal(model.centers_, [[offset]])
    assert np.sum((X - model.centers_[0]) ** 2) == 2.0
    np.testing.assert_allclose(model.cov_inv, [[[1.0 / (1.0 + 1e-6)]]], rtol=2e-15)


@pytest.mark.parametrize("scale", [1.0, 1e-14, 0.0])
def test_scatter_normalizes_positive_support_before_ridge(monkeypatch, scale):
    X = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    model = SoftDBSCANGM()
    model.centers = np.zeros((2, 2))
    weights = np.column_stack([np.full(4, scale), np.zeros(4)])
    monkeypatch.setattr(model, "_chunk_rows", lambda *args: 2)
    with np.errstate(divide="raise", invalid="raise"):
        actual = model._covariances(X, weights, weights.sum(axis=0))
    expected = np.zeros((2, 2, 2))
    if scale > 0:
        expected[0] = np.diag([2.0, 0.5])
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("far_distance", [1e7, 1e200])
def test_iterative_centers_keep_tiny_positive_and_zero_support(
    monkeypatch, far_distance
):
    X = np.array([[8.0, 0.0], [9.0, 0.0], [11.0, 0.0], [12.0, 0.0]])
    model = SoftDBSCANGM(eps=1.1, min_samples=1, max_iter=2, tol=0.0)
    # Prescribe finite distances to exercise the next completed mean/scatter
    # block with tiny positive or exactly underflowed membership support.
    monkeypatch.setattr(
        model,
        "_mahalanobis_distances",
        lambda data, cov: np.tile([1.0, far_distance], (len(data), 1)),
    )
    with np.errstate(divide="raise", invalid="raise"):
        model.fit(X)
    np.testing.assert_allclose(model.centers_[0], [10.0, 0.0], rtol=2e-15, atol=0)
    expected_cov = np.diag([2.5, 0.0]) + 1e-6 * np.eye(2)
    np.testing.assert_allclose(
        model.cov_inv[0], np.linalg.inv(expected_cov), rtol=2e-15
    )
    if far_distance == 1e7:
        assert np.all(model.U[:, 1] > 0)
        np.testing.assert_allclose(model.centers_[1], [10.0, 0.0], rtol=2e-15, atol=0)
        np.testing.assert_allclose(
            model.cov_inv[1], np.linalg.inv(expected_cov), rtol=2e-15
        )
    else:
        np.testing.assert_array_equal(model.U[:, 1], np.zeros(len(X)))
        np.testing.assert_array_equal(model.centers_[1], np.zeros(2))
        np.testing.assert_array_equal(model.cov_inv[1], 1e6 * np.eye(2))


@pytest.fixture
def X():
    rng = np.random.default_rng(19)
    return np.vstack(
        [rng.normal([0, 0], 0.3, (20, 2)), rng.normal([4, 4], 0.3, (20, 2))]
    )


def test_fit_runs(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    assert model.U is not None


def test_membership_shape(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    M = model.get_membership()
    assert M.ndim == 2
    assert M.shape[0] == 40


def test_membership_rows_sum_to_one(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    M = model.get_membership()
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-5)


def test_predict_shape(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    labels = model.predict()
    assert labels.shape == (40,)


def test_centers_shape(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    k = model.U.shape[1]
    assert model.centers.shape == (k, 2)


def test_labels_stored(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    assert model.labels_ is not None
    assert model.labels_.shape == (40,)


def test_canonical_centers_refresh_on_refit_and_remain_transductive():
    model = SoftDBSCANGM(eps=0.5, min_samples=2, max_iter=3)
    for locations in ([-2.0, 2.0], [-5.0, 0.0, 5.0]):
        X = np.array([[x + delta] for x in locations for delta in (0.0, 0.1)])
        assert model.fit(X) is model
        assert model.centers_.shape == (len(locations), 1)
        np.testing.assert_array_equal(model.centers_, model.centers)
        np.testing.assert_allclose(
            model.centers_[:, 0], np.array(locations) + 0.05, atol=1e-5
        )
        np.testing.assert_array_equal(model.predict_proba(), model.U)
        np.testing.assert_array_equal(model.predict(), model.U.argmax(axis=1))
        for method in (model.predict, model.predict_proba):
            with pytest.raises(NotImplementedError, match="transductive"):
                method(X)


def test_canonical_centers_enable_prototype_metrics():
    X = np.array([[-2.0], [-1.9], [2.0], [2.1]])
    model = SoftDBSCANGM(eps=0.5, min_samples=2, max_iter=3).fit(X)
    metrics = soft_clustering_metrics(X, model.memberships_, model.centers_)
    expected = {
        "fuzzy_hypervolume": 0.10000781463395918,
        "xie_beni": 0.00015622553255633192,
        "fuzzy_compactness": 0.009998437011528045,
        "fuzzy_separation": 4.000000585676229,
    }
    for name, value in expected.items():
        assert np.isfinite(metrics[name])
        np.testing.assert_allclose(metrics[name], value, rtol=1e-10, atol=1e-12)
