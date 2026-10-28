# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for AFCMSimple (adaptive fuzzy c-means without graph embedding)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from soft_clustering import AFCMSimple


@pytest.fixture
def blobs():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(c, 0.6, (60, 2)) for c in ([0, 0], [4, 0], [0, 4])])
    return X, np.repeat([0, 1, 2], 60)


def test_recovers_planted_clusters(blobs):
    X, y = blobs
    model = AFCMSimple(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.95


def test_learned_precision_matches_the_shared_variance(blobs):
    """gamma = 1 / (2 sigma^2) for isotropic clusters of standard deviation sigma."""
    X, _ = blobs
    gamma = AFCMSimple(n_clusters=3, random_state=0).fit(X).gamma_
    assert gamma == pytest.approx(1.0 / (2 * 0.6**2), rel=0.15)


def test_matches_a_spherical_tied_gaussian_mixture(blobs):
    """The model is EM for equal-weight isotropic Gaussians with shared variance."""
    X, _ = blobs
    ours = AFCMSimple(n_clusters=3, random_state=0, tol=1e-12, max_iter=500).fit(X)
    reference = GaussianMixture(
        3, covariance_type="tied", random_state=0, tol=1e-10
    ).fit(X)
    gap = np.linalg.norm(
        ours.centers_[:, None, :] - reference.means_[None, :, :], axis=2
    )
    assert gap.min(axis=1).max() < 0.1


def test_objective_is_non_increasing(blobs):
    X, _ = blobs
    J = AFCMSimple(n_clusters=3, random_state=0).fit(X).objective_trajectory_
    assert np.all(np.diff(J) <= 1e-9 * np.abs(J).max())


def test_out_of_sample_matches_training(blobs):
    X, _ = blobs
    model = AFCMSimple(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_allclose(model.predict_proba(X), model.memberships_)


def test_fuzzifier_was_removed():
    """The reference model has no fuzzifier; an `m` argument is an error."""
    with pytest.raises(TypeError):
        AFCMSimple(n_clusters=2, m=2.0)


def test_more_clusters_than_samples_raises():
    with pytest.raises(ValueError, match="exceeds the number of samples"):
        AFCMSimple(n_clusters=5).fit(np.zeros((3, 2)))
