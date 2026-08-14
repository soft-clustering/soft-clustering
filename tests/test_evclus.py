# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for EVCLUS (evidential clustering of proximity data)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import EVCLUS


def blobs(n_per=30, d=3, k=3, spread=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(3.0 * i, spread, (n_per, d)) for i in range(k)])
    return X, np.repeat(np.arange(k), n_per)


def pairwise(X):
    sq = (X**2).sum(axis=1)
    d = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    return np.sqrt(d)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    X, _ = blobs()
    model = EVCLUS(n_clusters=3, random_state=0)
    assert model.fit(X) is model


def test_output_shapes():
    X, _ = blobs()
    model = EVCLUS(n_clusters=3, random_state=0).fit(X)
    assert model.memberships_.shape == (X.shape[0], 3)
    assert model.masses_.shape == (X.shape[0], 4)


def test_credal_partition_is_a_valid_mass_assignment():
    X, _ = blobs()
    masses = EVCLUS(n_clusters=3, random_state=0).fit(X).masses_
    assert np.all(masses >= 0)
    np.testing.assert_allclose(masses.sum(axis=1), 1.0, atol=1e-9)


def test_pignistic_memberships_sum_to_one():
    X, _ = blobs()
    U = EVCLUS(n_clusters=3, random_state=0).fit(X).memberships_
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)


def test_betp_matches_the_pignistic_transform():
    """BetP_ik = m_ik + m_i(Omega)/K."""
    X, _ = blobs()
    model = EVCLUS(n_clusters=3, random_state=0).fit(X)
    expected = model.masses_[:, :3] + model.masses_[:, [3]] / 3
    np.testing.assert_allclose(model.betp(), expected)
    np.testing.assert_allclose(model.memberships_, expected)


def test_ignorance_is_the_omega_mass():
    X, _ = blobs()
    model = EVCLUS(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_allclose(model.ignorance(), model.masses_[:, -1])
    assert np.all(model.ignorance() >= 0) and np.all(model.ignorance() <= 1)


def test_labels_are_argmax():
    X, _ = blobs()
    model = EVCLUS(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(model.labels_, np.argmax(model.memberships_, axis=1))


def test_reproducible_under_fixed_seed():
    X, _ = blobs()
    first = EVCLUS(n_clusters=3, random_state=2).fit(X).memberships_
    second = EVCLUS(n_clusters=3, random_state=2).fit(X).memberships_
    np.testing.assert_allclose(first, second)


def test_accessors_require_a_fit():
    model = EVCLUS(n_clusters=3)
    with pytest.raises(RuntimeError):
        model.betp()
    with pytest.raises(RuntimeError):
        model.ignorance()


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


class TestObjective:
    def test_analytic_gradient_matches_finite_differences(self):
        """The stress gradient is derived by hand; this pins it down."""
        X, _ = blobs(n_per=8, k=2)
        model = EVCLUS(n_clusters=2)
        delta = model._dissimilarities(X)
        norm = 0.5 * float(np.sum(delta**2))

        rng = np.random.default_rng(0)
        alpha = rng.normal(size=(X.shape[0], 3)).ravel()
        _, gradient = model._objective(alpha, delta, norm)

        step = 1e-6
        for index in (0, 5, 11, 23, alpha.size - 1):
            bump = np.zeros_like(alpha)
            bump[index] = step
            numerical = (
                model._objective(alpha + bump, delta, norm)[0]
                - model._objective(alpha - bump, delta, norm)[0]
            ) / (2 * step)
            assert abs(numerical - gradient[index]) < 1e-5 * max(1.0, abs(numerical))

    def test_stress_is_low_on_separable_data(self):
        X, _ = blobs(seed=1)
        model = EVCLUS(n_clusters=3, random_state=0, n_init=3).fit(X)
        assert 0.0 <= model.stress_ < 0.05

    def test_more_restarts_do_not_worsen_the_stress(self):
        X, _ = blobs(seed=5)
        one = EVCLUS(n_clusters=3, n_init=1, random_state=0).fit(X).stress_
        many = EVCLUS(n_clusters=3, n_init=4, random_state=0).fit(X).stress_
        assert many <= one + 1e-12


# ---------------------------------------------------------------------------
# Clustering quality and input modes
# ---------------------------------------------------------------------------


def test_recovers_well_separated_blobs():
    X, y = blobs(seed=1)
    model = EVCLUS(n_clusters=3, random_state=0, n_init=3).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.95


def test_precomputed_matches_euclidean():
    """EVCLUS is defined on proximities; the feature path is a convenience."""
    X, _ = blobs(seed=1)
    from_features = EVCLUS(n_clusters=3, random_state=0).fit(X)
    from_distances = EVCLUS(n_clusters=3, metric="precomputed", random_state=0).fit(
        pairwise(X)
    )
    np.testing.assert_allclose(
        from_features.memberships_, from_distances.memberships_, atol=1e-6
    )


def test_ambiguous_points_carry_more_ignorance():
    """An outlier equidistant from every cluster should be less committed."""
    X, _ = blobs(n_per=25, d=2, k=3, seed=4)
    X = np.vstack([X, np.array([[3.0, 3.0]])])  # sits between the blobs
    model = EVCLUS(n_clusters=3, random_state=0, n_init=3).fit(X)
    assert model.ignorance()[-1] > np.median(model.ignorance()[:-1])


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_identical_points_give_total_ignorance(self):
        X = np.ones((10, 3))
        model = EVCLUS(n_clusters=3, random_state=0).fit(X)
        np.testing.assert_allclose(model.ignorance(), 1.0)
        np.testing.assert_allclose(model.memberships_, 1.0 / 3)

    def test_single_cluster(self):
        X, _ = blobs()
        U = EVCLUS(n_clusters=1, random_state=0).fit(X).memberships_
        np.testing.assert_allclose(U, 1.0)

    def test_duplicate_points_stay_finite(self):
        X, _ = blobs(n_per=5)
        X = np.repeat(X, 3, axis=0)
        U = EVCLUS(n_clusters=2, random_state=0).fit(X).memberships_
        assert np.isfinite(U).all()


@pytest.mark.parametrize("kwargs", [{"n_clusters": 0}, {"max_iter": 0}, {"n_init": 0}])
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        EVCLUS(**kwargs)


def test_invalid_metric_raises():
    with pytest.raises(ValueError, match="metric must be"):
        EVCLUS(metric="cosine")


def test_non_square_precomputed_raises():
    with pytest.raises(ValueError, match="square"):
        EVCLUS(n_clusters=2, metric="precomputed").fit(np.zeros((4, 6)))


def test_more_clusters_than_samples_raises():
    X, _ = blobs(n_per=1, k=2)
    with pytest.raises(ValueError, match="smaller than"):
        EVCLUS(n_clusters=10).fit(X)
