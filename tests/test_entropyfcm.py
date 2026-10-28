# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for ENTROPYFCM (entropy-regularised fuzzy c-means)."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from soft_clustering import ENTROPYFCM


@pytest.fixture
def blobs():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(c, 0.5, (50, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    return X, np.repeat([0, 1, 2], 50)


def test_entropy_weight_controls_fuzziness(blobs):
    """Regression: earlier releases ignored entropy_weight entirely."""
    X, _ = blobs
    crisp = ENTROPYFCM(n_clusters=3, entropy_weight=0.05, random_state=0).fit(X)
    fuzzy = ENTROPYFCM(n_clusters=3, entropy_weight=5.0, random_state=0).fit(X)
    assert (
        crisp.memberships_.max(axis=1).mean()
        > fuzzy.memberships_.max(axis=1).mean() + 0.1
    )


def test_memberships_are_the_gibbs_distribution(blobs):
    X, _ = blobs
    lam = 0.7
    model = ENTROPYFCM(n_clusters=3, entropy_weight=lam, random_state=0).fit(X)
    d2 = ((X[:, None, :] - model.centers_[None, :, :]) ** 2).sum(axis=2)
    expected = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / lam)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(model.memberships_, expected, atol=1e-10)


def test_small_weight_approaches_kmeans(blobs):
    X, _ = blobs
    ours = ENTROPYFCM(n_clusters=3, entropy_weight=1e-3, random_state=0).fit(X)
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(km.labels_, ours.labels_) == 1.0


def test_objective_is_non_increasing(blobs):
    X, _ = blobs
    J = ENTROPYFCM(n_clusters=3, random_state=0).fit(X).objective_trajectory_
    assert np.all(np.diff(J) <= 1e-9 * np.abs(J).max())


def test_recovers_planted_clusters(blobs):
    X, y = blobs
    model = ENTROPYFCM(n_clusters=3, entropy_weight=0.5, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.95


def test_out_of_sample_matches_training(blobs):
    X, _ = blobs
    model = ENTROPYFCM(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_allclose(model.predict_proba(X), model.memberships_)


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_non_positive_weight_raises(blobs, weight):
    X, _ = blobs
    with pytest.raises(ValueError, match="entropy_weight"):
        ENTROPYFCM(n_clusters=3, entropy_weight=weight).fit(X)
