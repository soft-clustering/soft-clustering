# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for SFCMEP (Semi-supervised FCM with Membership Prior)."""

import warnings

import numpy as np
import pytest

from soft_clustering import SFCMEP


@pytest.fixture
def Xy():
    rng = np.random.default_rng(31)
    X = np.vstack([rng.normal([0, 0], 0.4, (20, 2)), rng.normal([5, 5], 0.4, (20, 2))])
    # Label first 5 samples of each cluster; rest = None
    y = np.array([0] * 5 + [None] * 15 + [1] * 5 + [None] * 15, dtype=object)
    return X, y


def test_fit_predict_returns_the_membership_matrix(Xy):
    X, y = Xy
    model = SFCMEP(n_clusters=2, random_state=0)
    U = model.fit_predict(X, y)
    assert isinstance(U, np.ndarray)
    assert U.shape == (40, 2)
    np.testing.assert_array_equal(U, model.memberships_)
    assert model.centers_.shape == (2, 2)


def test_membership_nonneg(Xy):
    X, y = Xy
    U = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert np.all(U >= 0)


def test_membership_rows_sum_to_one(Xy):
    """Every sample must hold a full unit of membership. Underflow in the
    exponential weighting used to zero whole rows: a point further than a few
    multiples of ``lam`` from every centroid produced exp() == 0 for all
    clusters, and the 1e-12 denominator guard then collapsed the column."""
    X, y = Xy
    U = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert np.allclose(U.sum(axis=1), 1.0)
    assert not np.any(np.isclose(U.sum(axis=1), 0.0))


def test_membership_survives_a_large_coordinate_scale(Xy):
    """The same underflow, made explicit: scaling the data up while holding
    ``lam`` fixed must not destroy the partition."""
    X, y = Xy
    U = SFCMEP(K=2, random_state=0).fit_predict(X * 50.0, y)
    assert np.allclose(U.sum(axis=1), 1.0)


def test_centroids_are_not_degenerate(Xy):
    """Regression test for a missing summation axis in the centroid update.
    Summing the weighted samples over every axis instead of over samples only
    yields a scalar, which broadcasts into all D coordinates — the signature of
    the bug is a centroid whose components are all identical."""
    X, y = Xy
    model = SFCMEP(K=2, random_state=0)
    model.fit_predict(X, y)
    V = model.centers_

    for centre in V:
        assert not np.allclose(centre, centre[0]), (
            f"centroid {centre} has identical components in every dimension, "
            "which indicates the weighted mean collapsed to a scalar"
        )


def test_recovers_two_well_separated_clusters(Xy):
    """End-to-end sanity: with two clean blobs and a few labels per class, the
    partition should match the truth and the centroids should sit on the blobs."""
    X, y = Xy
    model = SFCMEP(K=2, random_state=0)
    model.fit_predict(X, y)

    truth = np.array([0] * 20 + [1] * 20)
    agreement = max(
        (model.labels_ == truth).mean(),
        (model.labels_ == 1 - truth).mean(),
    )
    assert agreement == 1.0

    centres = sorted(model.centers_.tolist(), key=lambda c: c[0])
    assert np.allclose(centres[0], [0, 0], atol=0.5)
    assert np.allclose(centres[1], [5, 5], atol=0.5)


def test_fully_unlabeled():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(20, 2))
    y = np.array([None] * 20, dtype=object)
    U = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert U.shape == (20, 2)


def test_fully_unlabeled_produces_a_real_partition():
    """With no labels there is no prior to estimate. Deriving one from an empty
    set used to make every entry NaN, which this test pins shut: the shape
    assertion above passed happily on an all-NaN matrix."""
    rng = np.random.default_rng(32)
    X = rng.normal(size=(20, 2))
    y = np.array([None] * 20, dtype=object)

    U = SFCMEP(K=2, random_state=0).fit_predict(X, y)

    assert not np.isnan(U).any()
    assert np.allclose(U.sum(axis=1), 1.0)


def test_fully_unlabeled_emits_no_warnings():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(20, 2))
    y = np.array([None] * 20, dtype=object)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        SFCMEP(K=2, random_state=0).fit_predict(X, y)


def test_populates_the_estimator_protocol(Xy):
    """The solution is published under the protocol attribute names."""
    X, y = Xy
    model = SFCMEP(K=2, random_state=0)
    model.fit_predict(X, y)

    assert model.memberships_.shape == (40, 2)
    assert model.labels_.shape == (40,)
    assert model.centers_.shape == (2, 2)
    assert model.n_clusters_ == 2
    assert np.array_equal(model.labels_, np.argmax(model.memberships_, axis=1))


def test_accepts_n_clusters_alias(Xy):
    X, y = Xy
    model = SFCMEP(n_clusters=2, random_state=0)
    model.fit_predict(X, y)
    assert model.memberships_.shape == (40, 2)


def test_temperature_controls_softness(Xy):
    """Regression: the prior used to compound across sweeps.

    Earlier releases re-weighted the previous sweep's memberships instead of
    the prior, so memberships collapsed to crisp assignments whatever ``lam``
    was. With the prior held fixed, a larger temperature gives softer
    memberships.
    """
    X, y = Xy
    sharp = SFCMEP(n_clusters=2, lam=0.5, random_state=0).fit_predict(X, y)
    soft = SFCMEP(n_clusters=2, lam=20.0, random_state=0).fit_predict(X, y)
    assert soft.max(axis=1).mean() < sharp.max(axis=1).mean() - 0.05


def test_memberships_are_the_prior_reweighted_by_distance(Xy):
    """u_ij is proportional to prior_ij * exp(-||x_j - v_i||^2 / lam)."""
    X, y = Xy
    lam = 3.0
    model = SFCMEP(n_clusters=2, lam=lam, random_state=0)
    U = model.fit_predict(X, y)
    d2 = ((X[:, None, :] - model.centers_[None, :, :]) ** 2).sum(axis=2)
    prior = np.maximum(model.prior_memberships_, np.finfo(np.float64).tiny)
    expected = prior * np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / lam)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(U, expected, atol=1e-6)


@pytest.mark.parametrize("unlabelled", [None, -1, np.nan])
def test_unlabelled_markers_are_equivalent(Xy, unlabelled):
    X, y = Xy
    y_marker = np.array([v if v is not None else unlabelled for v in y], dtype=object)
    reference = SFCMEP(n_clusters=2, random_state=0).fit_predict(X, y)
    np.testing.assert_allclose(
        SFCMEP(n_clusters=2, random_state=0).fit_predict(X, y_marker), reference
    )


def test_label_out_of_range_raises(Xy):
    X, y = Xy
    y = y.copy()
    y[0] = 5
    with pytest.raises(ValueError, match="labels must be integers"):
        SFCMEP(n_clusters=2).fit_predict(X, y)


def test_refit_is_reproducible(Xy):
    X, y = Xy
    model = SFCMEP(n_clusters=2, random_state=4)
    first = model.fit_predict(X, y).copy()
    np.testing.assert_array_equal(model.fit_predict(X, y), first)


def _three_blobs():
    rng = np.random.default_rng(0)
    centers = ([0.0, 0.0], [3.0, 0.0], [0.0, 3.0])
    X = np.vstack([rng.normal(c, 0.6, (50, 2)) for c in centers])
    return X, np.repeat([0, 1, 2], 50)


def test_one_label_per_class_gives_a_finite_partition():
    """Regression: a single labelled sample is its class's initial centroid,
    so its distance to it is zero. The prior computed ``d_min / d`` as 0/0,
    and the resulting NaN spread silently to every membership."""
    X, truth = _three_blobs()
    y = np.full(X.shape[0], -1)
    y[[0, 50, 100]] = [0, 1, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = SFCMEP(n_clusters=3, random_state=0).fit(X, y)
    assert np.all(np.isfinite(model.prior_memberships_))
    assert np.all(np.isfinite(model.memberships_))
    np.testing.assert_array_equal(model.labels_[[0, 50, 100]], [0, 1, 2])


def test_label_enforcement_is_monotone_in_rho():
    """Regression: a zero prior entry (the other classes of a labelled sample
    at rho = 1) was replaced by one, the largest weight, so rho = 1 enforced
    the labels less than rho = 0.9."""
    X, truth = _three_blobs()
    rng = np.random.default_rng(1)
    idx = rng.choice(X.shape[0], size=30, replace=False)
    y = np.full(X.shape[0], -1)
    y[idx] = truth[idx]
    own = [
        SFCMEP(n_clusters=3, random_state=0, rho=rho)
        .fit(X, y)
        .memberships_[idx, truth[idx]]
        .mean()
        for rho in (0.5, 0.9, 1.0)
    ]
    assert own[0] <= own[1] <= own[2]


def test_conflicting_prior_and_distance_keep_normalized_memberships():
    X = np.array([0.0, 100.0, 1.0, 2.0])[:, None]
    model = SFCMEP(n_clusters=2, rho=1.0, lam=1.0, max_iter=20, random_state=0).fit(
        X, np.array([0, 0, 1, 1])
    )
    assert np.isfinite(model.memberships_).all()
    assert (model.memberships_ >= 0).all()
    np.testing.assert_allclose(model.memberships_.sum(1), 1, rtol=0, atol=1e-15)


@pytest.mark.parametrize("near_prior", [0.0, 1e-20])
def test_full_prior_weighted_score_retains_zero_policy_and_tiny_positive_prior(
    near_prior,
):
    model = SFCMEP(n_clusters=2, lam=1.0)
    model.N = 1
    actual = model._update_membership_matrix(
        np.array([[0.0]]), np.array([[1.0], [near_prior]]), np.array([[10.0], [0.0]])
    )
    weights = np.array([np.exp(-100.0), max(near_prior, np.finfo(np.float64).tiny)])
    expected = weights / weights.sum()
    np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-12, atol=0)
    assert actual.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("lam", [0.0, 1e-320])
def test_zero_and_tiny_temperatures_keep_assignment_and_tie_contract(lam):
    model = SFCMEP(n_clusters=2, lam=lam)
    model.N = 3
    prior = np.array([[0.0, 0.25, 1.0], [1.0, 0.75, 0.0]])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        actual = model._update_membership_matrix(
            np.array([[0.0], [1.0], [2.0]]), prior, np.array([[0.0], [2.0]])
        )
    np.testing.assert_array_equal(actual[:, [0, 2]], np.eye(2))
    np.testing.assert_allclose(actual[:, 1], [1.0, 0.0] if lam == 0 else [0.25, 0.75])
    np.testing.assert_allclose(actual.sum(0), 1)
