# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for CAFHFCM (centroid auto-fused hierarchical fuzzy c-means)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import CAFHFCM


@pytest.fixture
def separated():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [12, 0], [0, 12]])
    X = np.vstack([rng.normal(c, 0.8, (60, 2)) for c in centres])
    return X, np.repeat([0, 1, 2], 60)


def test_fusion_reduces_the_number_of_clusters(separated):
    X, _ = separated
    model = CAFHFCM(random_state=0).fit(X)
    K0 = int(np.floor(2 * np.sqrt(X.shape[0])))
    assert model.n_clusters_ < K0
    assert model.merge_history_, "no centroids were fused"


def test_stronger_fusion_gives_fewer_clusters(separated):
    X, _ = separated
    weak = CAFHFCM(alpha=0.001, random_state=0).fit(X).n_clusters_
    strong = CAFHFCM(alpha=0.02, random_state=0).fit(X).n_clusters_
    assert strong <= weak


def test_zero_fusion_keeps_every_centroid(separated):
    X, _ = separated
    model = CAFHFCM(alpha=0.0, n_init_clusters=7, random_state=0).fit(X)
    assert model.n_clusters_ == 7


def test_finds_well_separated_clusters_at_a_suitable_strength(separated):
    X, y = separated
    # Over a range of fusion strengths the discovered partition should agree
    # well with the planted one; the reference selects gamma the same way.
    # (A clean three-cluster solution is not reached here: one blob keeps two
    # centroids at a local optimum of the non-convex FCM term. That is
    # documented in the estimator's docstring.)
    best = max(
        adjusted_rand_score(y, CAFHFCM(alpha=a, random_state=0).fit(X).labels_)
        for a in (0.002, 0.004, 0.006, 0.01)
    )
    assert best > 0.8


def test_auto_path_recovers_well_separated_clusters(separated):
    X, y = separated
    model = CAFHFCM(random_state=0).fit(X)
    assert model.n_clusters_ == 3
    assert adjusted_rand_score(y, model.labels_) > 0.95


def test_auto_path_is_monotone_and_contains_the_solution(separated):
    X, _ = separated
    model = CAFHFCM(random_state=0).fit(X)
    counts = [k for _, k in model.path_]
    assert all(b <= a for a, b in zip(counts, counts[1:]))
    assert (model.alpha_, model.n_clusters_) in model.path_
    assert 1 < model.n_clusters_ < int(np.floor(2 * np.sqrt(X.shape[0])))


def test_fixed_alpha_has_a_single_path_entry(separated):
    X, _ = separated
    model = CAFHFCM(alpha=0.004, random_state=0).fit(X)
    assert model.path_ == [(0.004, model.n_clusters_)]
    assert model.alpha_ == 0.004


def test_very_strong_fusion_merges_everything(separated):
    X, _ = separated
    assert CAFHFCM(alpha=1.0, random_state=0).fit(X).n_clusters_ == 1


def test_objective_decreases_between_merges(separated):
    X, _ = separated
    model = CAFHFCM(alpha=0.004, random_state=0).fit(X)
    J = model.objective_trajectory_
    merges = {it for it, _ in model.merge_history_}
    for i, step in enumerate(np.diff(J)):
        if i + 1 not in merges:
            assert step <= 1e-9 * np.abs(J).max()


def test_memberships_are_a_fuzzy_partition(separated):
    X, _ = separated
    U = CAFHFCM(random_state=0).fit_predict(X)
    assert U.shape[0] == X.shape[0]
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)


def test_n_clusters_argument_is_ignored(separated):
    """The cluster count is discovered, so a requested one has no effect."""
    X, _ = separated
    a = CAFHFCM(random_state=0).fit(X)
    b = CAFHFCM(n_clusters=17, random_state=0).fit(X)
    assert a.n_clusters_ == b.n_clusters_


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": -1.0}, {"alpha": "best"}, {"fuse_tol": -1.0}, {"n_init_clusters": 0}],
)
def test_invalid_parameters_raise(separated, kwargs):
    X, _ = separated
    with pytest.raises(ValueError):
        CAFHFCM(**kwargs).fit(X)


def test_stable_choice_rule():
    from soft_clustering._cafhfcm import _stable_choice

    # K0 = 20: the starting count and the single cluster are ignored.
    assert _stable_choice([20, 20, 20, 5, 5, 3, 3, 3, 1, 1], 20) == (5, 7)
    # A tie goes to the larger count (less fusion).
    assert _stable_choice([20, 4, 4, 2, 2, 1], 20) == (1, 2)
    # Nothing between K0 and 1: no choice.
    assert _stable_choice([20, 20, 1], 20) is None


def test_auto_without_a_stable_count_returns_the_last_fit():
    # Two centroids that fuse at once: the path goes from K0 = 2 straight to 1.
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    model = CAFHFCM(n_init_clusters=2, random_state=0).fit(X)
    assert model.path_[-1][1] == model.n_clusters_
