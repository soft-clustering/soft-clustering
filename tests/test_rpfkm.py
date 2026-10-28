# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for RPFKM (robust projected fuzzy k-means)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import RPFKM


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
