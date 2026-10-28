# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Shared tests for the soft consensus estimators SCSPA, SHBGF and SMCLA."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

import soft_clustering as sc

CONSENSUS = ["SCSPA", "SHBGF", "SMCLA"]


def noisy_ensemble(truth, n_partitions=4, k=3, noise=0.15, seed=0):
    """Soft partitions that agree with ``truth`` up to a label permutation."""
    rng = np.random.default_rng(seed)
    ensemble = []
    for q in range(n_partitions):
        perm = rng.permutation(k)
        U = rng.random((truth.size, k)) * noise
        U[np.arange(truth.size), perm[truth]] += 1.0
        ensemble.append(U / U.sum(axis=1, keepdims=True))
    return ensemble


@pytest.fixture
def truth():
    return np.repeat([0, 1, 2], 25)


@pytest.mark.parametrize("name", CONSENSUS)
def test_recovers_the_consensus_under_label_permutations(name, truth):
    model = getattr(sc, name)(n_clusters=3, random_state=0)
    model.fit_predict(noisy_ensemble(truth))
    assert adjusted_rand_score(truth, model.labels_) == 1.0


@pytest.mark.parametrize("name", CONSENSUS)
def test_idempotent_on_a_single_crisp_partition(name, truth):
    """The consensus of one crisp partition is that partition."""
    onehot = np.eye(3)[truth]
    model = getattr(sc, name)(n_clusters=3, random_state=0)
    model.fit_predict([onehot])
    assert adjusted_rand_score(truth, model.labels_) == 1.0


@pytest.mark.parametrize("name", CONSENSUS)
def test_inputs_are_not_modified(name, truth):
    ensemble = noisy_ensemble(truth)
    copies = [U.copy() for U in ensemble]
    getattr(sc, name)(n_clusters=3, random_state=0).fit_predict(ensemble)
    for U, V in zip(ensemble, copies):
        np.testing.assert_array_equal(U, V)


@pytest.mark.parametrize("name", CONSENSUS)
def test_reproducible(name, truth):
    ensemble = noisy_ensemble(truth, noise=0.8)
    a = getattr(sc, name)(n_clusters=3, random_state=5).fit_predict(ensemble)
    b = getattr(sc, name)(n_clusters=3, random_state=5).fit_predict(ensemble)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", CONSENSUS)
def test_mismatched_sample_counts_raise(name):
    with pytest.raises(ValueError, match="same samples"):
        getattr(sc, name)(n_clusters=2).fit_predict([np.ones((5, 2)), np.ones((6, 2))])


@pytest.mark.parametrize("name", CONSENSUS)
def test_negative_memberships_raise(name):
    with pytest.raises(ValueError, match="negative"):
        getattr(sc, name)(n_clusters=2).fit_predict([-np.ones((5, 2))])


@pytest.mark.parametrize("name", CONSENSUS)
def test_empty_ensemble_raises(name):
    with pytest.raises(ValueError, match="empty"):
        getattr(sc, name)(n_clusters=2).fit_predict([])


@pytest.mark.parametrize("name", ["SCSPA", "SHBGF"])
def test_crisp_consensus_is_one_hot(name, truth):
    U = getattr(sc, name)(n_clusters=3, random_state=0).fit_predict(
        noisy_ensemble(truth)
    )
    assert set(np.unique(U)) == {0.0, 1.0}


def test_smcla_consensus_is_soft(truth):
    """Regression: sMCLA's soft consensus used to be replaced by its arg-max."""
    U = sc.SMCLA(n_clusters=3, random_state=0).fit_predict(
        noisy_ensemble(truth, noise=0.6)
    )
    assert np.unique(U.round(6)).size > 3
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-12)
