# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for RoughKMeans (rough k-means, Lingras and West 2004)."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from soft_clustering import RoughKMeans


@pytest.fixture
def X():
    rng = np.random.default_rng(17)
    return np.vstack(
        [rng.normal([0, 0], 0.4, (20, 2)), rng.normal([5, 5], 0.4, (20, 2))]
    )


def test_approximations_are_published(X):
    model = RoughKMeans(n_clusters=2, random_state=0)
    U = model.fit_predict(X)
    assert U.shape == (40, 2)
    assert model.lower_approx_.shape == (40, 2)
    assert model.upper_approx_.shape == (40, 2)
    assert model.centers_.shape == (2, 2)
    assert model.n_iter_ >= 1


def test_rough_set_axioms(X):
    """Lower within upper; at most one lower approximation per object; every
    object in at least one upper approximation (Lingras and West, 2004)."""
    model = RoughKMeans(n_clusters=3, threshold=1.4, random_state=0).fit(X)
    L, U = model.lower_approx_, model.upper_approx_
    assert np.all(L <= U)
    assert np.all(L.sum(axis=1) <= 1)
    assert np.all(U.sum(axis=1) >= 1)
    # A boundary object lies in no lower approximation and >= 2 upper ones.
    boundary = L.sum(axis=1) == 0
    assert np.all(U[boundary].sum(axis=1) >= 2)


def test_threshold_one_reduces_to_kmeans(X):
    model = RoughKMeans(n_clusters=2, threshold=1.0, random_state=0).fit(X)
    assert np.all(model.lower_approx_.sum(axis=1) == 1)
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(km.labels_, model.labels_) == 1.0


def test_memberships_share_boundary_objects_equally(X):
    model = RoughKMeans(n_clusters=3, threshold=1.5, random_state=0).fit(X)
    U = model.memberships_
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    boundary = model.lower_approx_.sum(axis=1) == 0
    for row, upper in zip(U[boundary], model.upper_approx_[boundary]):
        np.testing.assert_allclose(row[upper == 1], 1.0 / upper.sum())


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="exceeds the number of samples"):
        RoughKMeans(n_clusters=3).fit(np.zeros((2, 2)))


@pytest.mark.parametrize("kwargs", [{"weight_lower": 1.5}, {"threshold": 0.5}])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        RoughKMeans(n_clusters=2, **kwargs)
