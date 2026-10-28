# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for FCC (fuzzy color clustering)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import FCC
from soft_clustering._fcc import _fcc_memberships


@pytest.fixture
def colours():
    """Three tight colour groups in CIELAB-like coordinates."""
    rng = np.random.default_rng(14)
    centres = np.array([[30.0, 10.0, -20.0], [60.0, -30.0, 40.0], [85.0, 5.0, 5.0]])
    X = np.vstack([rng.normal(c, 3.0, (40, 3)) for c in centres])
    return X, np.repeat([0, 1, 2], 40)


def test_membership_function_matches_the_fuzzy_colour_model():
    """Eq. 7 of Kim and Lee: 1 inside the JND ball, 0 for the others there,
    and the inverse-ratio rule on surface distances outside."""
    centres = np.array([[0.0, 0.0], [10.0, 0.0]])
    jnd = 2.0
    X = np.array([[1.0, 0.0], [5.0, 0.0], [4.0, 0.0]])
    U = _fcc_memberships(X, centres, jnd)
    np.testing.assert_allclose(U[0], [1.0, 0.0])
    np.testing.assert_allclose(U[1], [0.5, 0.5])
    # delta = (4 - 2, 6 - 2) = (2, 4) -> u = [1/2, 1/4] / (3/4)
    np.testing.assert_allclose(U[2], [2 / 3, 1 / 3])


def test_recovers_colour_groups_with_a_perceptual_radius(colours):
    X, y = colours
    model = FCC(n_clusters=3, jnd=8.0, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) == 1.0


def test_core_colours_have_crisp_membership(colours):
    X, _ = colours
    model = FCC(n_clusters=3, jnd=8.0, random_state=0).fit(X)
    d = np.linalg.norm(X[:, None, :] - model.centers_[None, :, :], axis=2)
    inside = (d <= 8.0).any(axis=1)
    assert inside.any()
    np.testing.assert_allclose(model.memberships_[inside].max(axis=1), 1.0)


def test_memberships_are_a_fuzzy_partition(colours):
    X, _ = colours
    U = FCC(n_clusters=3, random_state=0).fit_predict(X)
    assert U.shape == (120, 3)
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-12)


def test_out_of_sample_matches_training(colours):
    X, _ = colours
    model = FCC(n_clusters=3, jnd=8.0, random_state=0).fit(X)
    np.testing.assert_allclose(model.predict_proba(X), model.memberships_)


def test_reproducible(colours):
    X, _ = colours
    a = FCC(n_clusters=3, random_state=1).fit(X).memberships_
    b = FCC(n_clusters=3, random_state=1).fit(X).memberships_
    np.testing.assert_array_equal(a, b)


def test_negative_jnd_raises(colours):
    X, _ = colours
    with pytest.raises(ValueError, match="jnd"):
        FCC(n_clusters=3, jnd=-1.0).fit(X)
