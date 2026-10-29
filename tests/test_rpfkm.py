# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for RPFKM (robust projected fuzzy k-means)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import RPFKM


def test_zero_support_cluster_keeps_fit_finite():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    model = RPFKM(n_clusters=3, d=1, gamma=1e-6, max_iter=5, random_state=0)
    W, U, p = model._initialize_variables(X.T)
    M = model._update_M(X.T, W, U, p)
    first_U = model._update_U(X.T, W, M)
    assert np.any(first_U.sum(axis=1) == 0.0)
    with np.errstate(divide="raise", invalid="raise"):
        model.fit(X)
    assert np.isfinite(model.memberships_).all()
    assert np.all(model.memberships_ >= 0)
    np.testing.assert_allclose(model.memberships_.sum(axis=1), 1.0, atol=1e-14)
    assert np.isfinite(model.projection_).all()
    np.testing.assert_allclose(
        model.projection_.T @ model.projection_, np.eye(1), atol=1e-14
    )


@pytest.mark.parametrize("scale", [1.0, 1e-20])
def test_projection_scatter_omits_only_zero_mass(monkeypatch, scale):
    X = np.array([[10.0, 11.0, 14.0], [3.0, 7.0, 4.0]])
    U = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]])
    p = scale * np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 4.0]])
    weights = p[1] * U[1]
    center = np.average(X, axis=1, weights=weights)
    scatter = sum(w * np.outer(x - center, x - center) for w, x in zip(weights, X.T))
    original_eigh = np.linalg.eigh
    _, vectors = original_eigh(-scatter)
    expected_projection = vectors[:, -1:]
    observed = []

    def eigh(matrix):
        observed.append(matrix.copy())
        return original_eigh(matrix)

    monkeypatch.setattr(np.linalg, "eigh", eigh)
    with np.errstate(divide="raise", invalid="raise"):
        actual = RPFKM(n_clusters=2, d=1, beta=0.0)._update_W(X, U, p)
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0], -scatter, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        actual @ actual.T, expected_projection @ expected_projection.T, atol=1e-14
    )


@pytest.fixture
def X():
    """Two clusters separated along one axis, plus noise dimensions."""
    rng = np.random.default_rng(15)
    signal = np.concatenate([rng.normal(-4, 0.5, 40), rng.normal(4, 0.5, 40)])
    noise = rng.normal(0, 0.5, (80, 5))
    return np.column_stack([signal, noise]), np.repeat([0, 1], 40)


def test_fit_predict_returns_memberships(X):
    data, _ = X
    model = RPFKM(n_clusters=2, d=2, random_state=0)
    U = model.fit_predict(data)
    assert U.shape == (80, 2)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(U >= 0)


def test_projection_is_orthonormal(X):
    data, _ = X
    W = RPFKM(n_clusters=2, d=2, random_state=0).fit(data).projection_
    assert W.shape == (6, 2)
    np.testing.assert_allclose(W.T @ W, np.eye(2), atol=1e-8)


def test_recovers_clusters_in_the_projection(X):
    data, y = X
    model = RPFKM(n_clusters=2, d=1, random_state=0).fit(data)
    assert adjusted_rand_score(y, model.labels_) > 0.9


def test_reproducible(X):
    data, _ = X
    a = RPFKM(n_clusters=2, d=2, random_state=7).fit(data).memberships_
    b = RPFKM(n_clusters=2, d=2, random_state=7).fit(data).memberships_
    np.testing.assert_array_equal(a, b)


def test_legacy_c_keyword(X):
    data, _ = X
    assert RPFKM(c=3, d=2, random_state=0).fit(data).memberships_.shape == (80, 3)


@pytest.mark.parametrize("d", [0, 7])
def test_invalid_projection_dimension_raises(X, d):
    data, _ = X
    with pytest.raises(ValueError, match="d must lie"):
        RPFKM(n_clusters=2, d=d).fit(data)
