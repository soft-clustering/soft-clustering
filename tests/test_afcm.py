# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for AFCM (adaptive fuzzy c-means with graph embedding)."""

import numpy as np
import pytest
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

from soft_clustering import AFCM, FCM, _afcm


@pytest.fixture
def X():
    rng = np.random.default_rng(11)
    return np.vstack(
        [rng.normal([0, 0], 0.4, (20, 2)), rng.normal([5, 5], 0.4, (20, 2))]
    )


def test_fit_predict_returns_memberships(X):
    model = AFCM(n_clusters=2, n_neighbors=4, random_state=0)
    U = model.fit_predict(X)
    assert U.shape == (40, 2)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(U >= 0)


def test_embedding_shape(X):
    model = AFCM(n_clusters=3, n_neighbors=4, random_state=0).fit(X)
    assert model.embedding_.shape == (40, 3)
    # The embedding has orthonormal columns (the constraint X~ X~^T = I).
    np.testing.assert_allclose(
        model.embedding_.T @ model.embedding_, np.eye(3), atol=1e-8
    )


@pytest.mark.parametrize(
    "data",
    [
        make_moons(200, noise=0.05, random_state=0),
        make_circles(200, noise=0.03, factor=0.4, random_state=0),
    ],
    ids=["moons", "circles"],
)
def test_graph_term_separates_non_convex_clusters(data):
    """The point of the graph embedding: FCM cannot do this, AFCM can."""
    X, y = data
    afcm = AFCM(n_clusters=2, random_state=0).fit(X)
    fcm = FCM(n_clusters=2, random_state=0).fit(X)
    assert adjusted_rand_score(y, afcm.labels_) > 0.95
    assert adjusted_rand_score(y, fcm.labels_) < 0.5


def test_objective_is_non_increasing():
    X, _ = make_circles(150, noise=0.03, factor=0.4, random_state=1)
    J = AFCM(n_clusters=2, random_state=0).fit(X).objective_trajectory_
    # gamma is capped at a degenerate optimum, where the step is no longer an
    # exact minimiser; allow a relative 1e-5.
    assert np.all(np.diff(J) <= 1e-5 * np.abs(J).max())


def test_reproducible(X):
    a = AFCM(n_clusters=2, random_state=3).fit(X).memberships_
    b = AFCM(n_clusters=2, random_state=3).fit(X).memberships_
    np.testing.assert_array_equal(a, b)


def test_legacy_c_keyword_still_accepted(X):
    assert AFCM(c=2, random_state=0).fit(X).memberships_.shape == (40, 2)


@pytest.mark.parametrize(
    "kwargs", [{"lambda_": -1.0}, {"n_neighbors": 0}, {"sigma": 0.0}]
)
def test_invalid_parameters_raise(X, kwargs):
    with pytest.raises(ValueError):
        AFCM(n_clusters=2, **kwargs).fit(X)


@pytest.mark.parametrize(
    "X, requested_k, sigma",
    [
        (np.array([[0.0], [0.0], [0.0], [1.0]]), 1, 1.0),
        (np.zeros((4, 2)), 1, None),
        (np.zeros((4, 2)), 3, None),
        (np.zeros((4, 2)), 8, None),
        (np.array([[0.0], [2.0]]), 5, None),
    ],
    ids=["duplicates", "identical", "all-other-points", "capped", "two-points"],
)
def test_knn_graph_excludes_self_and_keeps_k_other_samples(
    monkeypatch, X, requested_k, sigma
):
    observed = []
    original = _afcm.NearestNeighbors.kneighbors

    def observe(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(_afcm.NearestNeighbors, "kneighbors", observe)
    laplacian, bandwidth = _afcm._normalized_knn_laplacian(X, requested_k, sigma)
    assert len(observed) == 1
    distances, indices = observed[0]
    n, k = len(X), min(requested_k, len(X) - 1)
    assert indices.shape == (n, k)
    for row, neighbors in enumerate(indices):
        assert row not in neighbors
        assert len(set(neighbors)) == k
        np.testing.assert_allclose(
            distances[row], np.linalg.norm(X[row] - X[neighbors], axis=1)
        )
    if not np.any(X):
        assert bandwidth == 1.0
    W = np.zeros((n, n))
    for row, neighbors in enumerate(indices):
        for column in neighbors:
            W[row, column] = np.exp(
                -np.sum((X[row] - X[column]) ** 2) / (2 * bandwidth**2)
            )
    W = np.maximum(W, W.T)
    degree = W.sum(axis=1)
    expected = np.eye(n) - W / np.sqrt(degree[:, None] * degree[None, :])
    np.testing.assert_allclose(laplacian, expected, atol=1e-15)
    np.testing.assert_array_equal(np.diag(laplacian), np.ones(n))


@pytest.mark.parametrize("sigma", [None, 2.0])
def test_untied_knn_laplacian_matches_independent_gaussian_graph(sigma):
    X = np.array([[0.0], [1.0], [4.0], [10.0]])
    # Nearest other points are 1, 0, 1, 2, with distances 1, 1, 3, 6.
    expected_sigma = 2.0 if sigma is None else sigma
    W = np.zeros((4, 4))
    for i, j, distance in [(0, 1, 1), (1, 0, 1), (2, 1, 3), (3, 2, 6)]:
        W[i, j] = W[j, i] = np.exp(-(distance**2) / (2 * expected_sigma**2))
    degree = W.sum(axis=1)
    expected = np.eye(4) - W / np.sqrt(degree[:, None] * degree[None, :])
    actual, actual_sigma = _afcm._normalized_knn_laplacian(X, 1, sigma)
    assert actual_sigma == expected_sigma
    np.testing.assert_allclose(actual, expected, atol=1e-15)
