# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for KFCCL (kernel-based fuzzy competitive learning)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel

from soft_clustering import KFCCL


@pytest.fixture
def X():
    rng = np.random.default_rng(6)
    return np.vstack(
        [rng.normal([0, 0], 0.5, (15, 2)), rng.normal([4, 4], 0.5, (15, 2))]
    )


def test_clusters_separate(X):
    """Regression: the previous implementation left every membership at 1/K."""
    model = KFCCL(n_clusters=2, gamma=0.2, lambda_=5.0, random_state=0).fit(X)
    assert adjusted_rand_score(np.repeat([0, 1], 15), model.labels_) == 1.0
    assert model.memberships_.max(axis=1).mean() > 0.6


def test_lambda_controls_fuzziness(X):
    soft = KFCCL(n_clusters=2, gamma=0.2, lambda_=1.0, random_state=0).fit(X)
    crisp = KFCCL(n_clusters=2, gamma=0.2, lambda_=20.0, random_state=0).fit(X)
    assert crisp.memberships_.max(axis=1).mean() > soft.memberships_.max(axis=1).mean()


def test_prototypes_are_convex_combinations_of_mapped_samples(X):
    """Every update keeps the expansion coefficients a convex combination."""
    G = KFCCL(n_clusters=2, gamma=0.2, random_state=0).fit(X).coefficients_
    assert np.all(G >= -1e-12)
    np.testing.assert_allclose(G.sum(axis=1), 1.0, atol=1e-10)


def test_memberships_match_the_kernel_trick_distances(X):
    model = KFCCL(n_clusters=2, gamma=0.2, lambda_=5.0, random_state=0).fit(X)
    K = rbf_kernel(X, gamma=0.2)
    G = model.coefficients_
    D = np.diag(K)[:, None] - 2 * (K @ G.T) + np.einsum("ij,jk,ik->i", G, K, G)[None, :]
    expected = np.exp(-5.0 * (D - D.min(axis=1, keepdims=True)))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(model.memberships_, expected, atol=1e-10)


def test_out_of_sample_matches_training(X):
    model = KFCCL(n_clusters=2, gamma=0.2, random_state=0).fit(X)
    np.testing.assert_allclose(model.predict_proba(X), model.memberships_, atol=1e-10)


def test_cluster_major_view_is_kept(X):
    model = KFCCL(n_clusters=2, random_state=0).fit(X)
    assert model.U.shape == (2, 30)
    np.testing.assert_array_equal(model.U, model.memberships_.T)


@pytest.mark.parametrize(
    "kwargs", [{"lambda_": 0.0}, {"gamma": -1.0}, {"learning_rate": 0.0}]
)
def test_invalid_parameters_raise(X, kwargs):
    with pytest.raises(ValueError):
        KFCCL(n_clusters=2, **kwargs).fit(X)
