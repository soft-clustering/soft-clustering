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


def test_fit_predict_returns_dict(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert isinstance(result, dict)


def test_dict_keys(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert "centroids" in result
    assert "membership_matrix" in result


def test_membership_matrix_shape(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert result["membership_matrix"].shape == (40, 2)


def test_centroids_shape(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert result["centroids"].shape == (2, 2)


def test_membership_nonneg(Xy):
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert np.all(result["membership_matrix"] >= 0)


def test_membership_rows_sum_to_one(Xy):
    """Every sample must hold a full unit of membership. Underflow in the
    exponential weighting used to zero whole rows: a point further than a few
    multiples of ``lam`` from every centroid produced exp() == 0 for all
    clusters, and the 1e-12 denominator guard then collapsed the column."""
    X, y = Xy
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    U = result["membership_matrix"]
    assert np.allclose(U.sum(axis=1), 1.0)
    assert not np.any(np.isclose(U.sum(axis=1), 0.0))


def test_membership_survives_a_large_coordinate_scale(Xy):
    """The same underflow, made explicit: scaling the data up while holding
    ``lam`` fixed must not destroy the partition."""
    X, y = Xy
    U = SFCMEP(K=2, random_state=0).fit_predict(X * 50.0, y)["membership_matrix"]
    assert np.allclose(U.sum(axis=1), 1.0)


def test_centroids_are_not_degenerate(Xy):
    """Regression test for a missing summation axis in the centroid update.
    Summing the weighted samples over every axis instead of over samples only
    yields a scalar, which broadcasts into all D coordinates — the signature of
    the bug is a centroid whose components are all identical."""
    X, y = Xy
    V = SFCMEP(K=2, random_state=0).fit_predict(X, y)["centroids"]

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
    result = SFCMEP(K=2, random_state=0).fit_predict(X, y)
    assert result["membership_matrix"].shape == (20, 2)


def test_fully_unlabeled_produces_a_real_partition():
    """With no labels there is no prior to estimate. Deriving one from an empty
    set used to make every entry NaN, which this test pins shut: the shape
    assertion above passed happily on an all-NaN matrix."""
    rng = np.random.default_rng(32)
    X = rng.normal(size=(20, 2))
    y = np.array([None] * 20, dtype=object)

    U = SFCMEP(K=2, random_state=0).fit_predict(X, y)["membership_matrix"]

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
    """SFCMEP returns a dict, which the base class cannot introspect; it must
    therefore publish its solution under the protocol attribute names."""
    X, y = Xy
    model = SFCMEP(K=2, random_state=0)
    model.fit_predict(X, y)

    assert model.memberships_.shape == (40, 2)
    assert model.labels_.shape == (40,)
    assert model.centers_.shape == (2, 2)
    assert model.n_clusters == 2
    assert np.array_equal(model.labels_, np.argmax(model.memberships_, axis=1))


def test_accepts_n_clusters_alias(Xy):
    X, y = Xy
    model = SFCMEP(n_clusters=2, random_state=0)
    model.fit_predict(X, y)
    assert model.memberships_.shape == (40, 2)
