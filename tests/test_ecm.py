# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for ECM (evidential c-means, Masson and Denoeux 2008).

Agreement with the authors' reference implementation (evclust) is checked in
``tests/test_external_agreement.py``.
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import ECM


@pytest.fixture
def X():
    rng = np.random.default_rng(7)
    return np.vstack(
        [rng.normal([0, 0], 0.5, (20, 2)), rng.normal([5, 5], 0.5, (20, 2))]
    )


def test_full_frame_has_every_non_empty_subset(X):
    model = ECM(n_clusters=3, random_state=0).fit(X)
    assert model.focal_sets_.shape == (7, 3)  # 2**3 - 1
    assert len({tuple(r) for r in model.focal_sets_}) == 7
    assert model.masses_.shape == (40, 8)  # empty set + 7 focal sets


def test_simple_frame(X):
    model = ECM(n_clusters=3, focal_sets="simple", random_state=0).fit(X)
    assert model.focal_sets_.shape == (4, 3)  # singletons + Omega
    assert model.masses_.shape == (40, 5)


def test_masses_form_a_mass_function(X):
    M = ECM(n_clusters=2, random_state=0).fit(X).masses_
    assert np.all(M >= 0)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)


def test_memberships_are_the_pignistic_transform(X):
    model = ECM(n_clusters=2, random_state=0).fit(X)
    F = model.focal_sets_.astype(float)
    m = model.masses_
    betp = (m[:, 1:] / F.sum(axis=1)) @ F / (1.0 - m[:, :1])
    np.testing.assert_allclose(model.memberships_, betp, atol=1e-10)


def test_objective_is_non_increasing(X):
    J = ECM(n_clusters=3, random_state=0).fit(X).objective_trajectory_
    assert np.all(np.diff(J) <= 1e-9 * np.abs(J).max())


def test_recovers_two_clusters(X):
    labels = ECM(n_clusters=2, random_state=0).fit(X).labels_
    assert adjusted_rand_score(np.repeat([0, 1], 20), labels) == 1.0


def test_ambiguous_point_puts_mass_on_the_pair(X):
    """A sample midway between two clusters is credited to {w1, w2}."""
    model = ECM(n_clusters=2, random_state=0, delta=20.0).fit(X)
    midpoint = model.centers_.mean(axis=0, keepdims=True)
    D, card = model._distances(midpoint, model.centers_)
    pair = int(np.flatnonzero(card == 2)[0])
    from soft_clustering._ecm import _ecm_masses

    M, _ = _ecm_masses(D, card, model.alpha, model.m, model.delta**2)
    assert np.argmax(M[0]) == pair


def test_far_outlier_goes_to_the_empty_set(X):
    model = ECM(n_clusters=2, delta=3.0, random_state=0).fit(X)
    outlier = np.array([[100.0, -100.0]])
    from soft_clustering._ecm import _ecm_masses

    D, card = model._distances(outlier, model.centers_)
    _, empty = _ecm_masses(D, card, model.alpha, model.m, model.delta**2)
    assert empty[0] > 0.99


def test_full_frame_is_refused_for_many_clusters():
    X = np.random.default_rng(0).normal(size=(40, 2))
    with pytest.raises(ValueError, match="focal_sets='simple'"):
        ECM(n_clusters=13).fit(X)


def test_backward_compatible_accessors(X):
    model = ECM(n_clusters=2, random_state=0).fit(X)
    np.testing.assert_array_equal(model.mass, model.masses_)
    np.testing.assert_array_equal(model.get_membership(), model.masses_)
    np.testing.assert_array_equal(model.prototypes, model.centers_)
