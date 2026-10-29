# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for FeMIFuzzy (Federated Fuzzy Clustering with MI)."""

from itertools import permutations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import FeMIFuzzy


@pytest.fixture
def clients_features():
    rng = np.random.default_rng(39)
    features_all = ["f1", "f2", "f3", "f4"]
    clients = [
        rng.normal(size=(8, 4)),
        rng.normal(size=(8, 4)),
    ]
    features = [features_all, features_all]
    return clients, features


def test_fit_predict_returns_the_global_memberships(clients_features):
    """fit_predict returns a membership matrix, not a list of centroids.

    Earlier releases returned only the global centroids and computed the
    memberships into a discarded expression, so the estimator produced no soft
    output at all.
    """
    clients, features = clients_features
    model = FeMIFuzzy(random_state=0, max_iter=20)
    result = model.fit_predict(clients, features)

    n_total = sum(len(c) for c in clients)
    assert isinstance(result, np.ndarray)
    assert result.shape == (n_total, model.n_clusters_)
    assert np.all(result >= 0)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(result, model.memberships_)


def test_global_centroids_shape(clients_features):
    clients, features = clients_features
    model = FeMIFuzzy(random_state=0, max_iter=20)
    model.fit_predict(clients, features)
    # Prototypes live in the space of the features the clients share.
    assert model.centers_.ndim == 2
    assert model.centers_.shape[1] == 4
    assert model.centers_.shape[0] == model.n_clusters_


def test_client_sizes_are_recorded(clients_features):
    clients, features = clients_features
    model = FeMIFuzzy(random_state=0, max_iter=20)
    model.fit_predict(clients, features)
    assert model.client_sizes_ == [len(c) for c in clients]


def test_reproducible_under_fixed_seed(clients_features):
    """The Sammon projection used to be seeded from a fresh unseeded RNG."""
    clients, features = clients_features
    first = FeMIFuzzy(random_state=0, max_iter=20).fit_predict(clients, features)
    second = FeMIFuzzy(random_state=0, max_iter=20).fit_predict(clients, features)
    np.testing.assert_allclose(first, second)


def test_sammon_mapping_preserves_distances():
    """A projection that ignores the data would make every downstream step noise."""
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(-3, 0.3, (10, 4)), rng.normal(3, 0.3, (10, 4))])
    projection = FeMIFuzzy(random_state=0)._sammon_mapping(X)

    high = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    low = np.linalg.norm(projection[:, None, :] - projection[None, :, :], axis=2)
    off = ~np.eye(len(X), dtype=bool)
    assert np.corrcoef(high[off], low[off])[0, 1] > 0.95


def test_recovers_planted_structure_across_clients():
    rng = np.random.default_rng(11)
    names = ["f0", "f1", "f2"]
    first = np.vstack([rng.normal(-3, 0.3, (10, 3)), rng.normal(3, 0.3, (10, 3))])
    second = np.vstack([rng.normal(-3, 0.3, (8, 3)), rng.normal(3, 0.3, (8, 3))])
    truth = np.concatenate([np.repeat([0, 1], 10), np.repeat([0, 1], 8)])

    model = FeMIFuzzy(random_state=0).fit([first, second], [names, names])
    assert adjusted_rand_score(truth, model.labels_) > 0.9


def test_no_common_features_raises():
    rng = np.random.default_rng(40)
    clients = [rng.normal(size=(5, 2)), rng.normal(size=(5, 2))]
    features = [["a", "b"], ["c", "d"]]
    model = FeMIFuzzy(random_state=0, max_iter=2)
    with pytest.raises(ValueError):
        model.fit_predict(clients, features)


def test_partial_overlap():
    rng = np.random.default_rng(41)
    clients = [rng.normal(size=(5, 3)), rng.normal(size=(5, 3))]
    features = [["a", "b", "c"], ["b", "c", "d"]]
    model = FeMIFuzzy(random_state=0, max_iter=10)
    result = model.fit_predict(clients, features)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 10


def test_single_sample_client_raises():
    """Xie-Beni model selection needs at least two prototypes per client."""
    rng = np.random.default_rng(42)
    clients = [rng.normal(size=(1, 3)), rng.normal(size=(5, 3))]
    features = [["a", "b", "c"], ["a", "b", "c"]]
    with pytest.raises(ValueError, match="at least 2 samples"):
        FeMIFuzzy(random_state=0, max_iter=5).fit_predict(clients, features)


@pytest.mark.parametrize("kwargs", [{"fuzzifier": 1.0}, {"n_imputations": 0}])
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        FeMIFuzzy(**kwargs)


@pytest.mark.parametrize("d", [1, 2, 5])
def test_empty_cluster_signature_uses_the_populated_summary_schema(d):
    X = np.array([np.arange(d), np.arange(d) + 2.0])
    U = np.array([[0.9, 0.1], [0.8, 0.2]])
    model = FeMIFuzzy()
    expected = np.concatenate(
        (
            [2],
            np.arange(d) + 1,
            np.arange(d),
            np.arange(d) + 2,
            np.ones(d),
            np.arange(d) + 1,
        )
    )
    np.testing.assert_array_equal(model._cluster_signature(X, U, 0), expected)
    np.testing.assert_array_equal(
        model._cluster_signature(X, U, 1), np.zeros(5 * d + 1)
    )


@pytest.mark.parametrize("empty_on_reference", [True, False])
def test_matching_with_an_empty_hard_cluster_is_a_minimum_cost_bijection(
    empty_on_reference,
):
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    empty_U = np.eye(3)[[0, 0, 1, 1]]
    full_U = np.eye(3)[[0, 1, 2, 2]]
    empty_signatures = np.array([[2, 1, 0, 2, 1, 1], [2, 11, 10, 12, 1, 11], [0] * 6])
    full_signatures = np.array(
        [[1, 0, 0, 0, 0, 0], [1, 2, 2, 2, 0, 2], [2, 11, 10, 12, 1, 11]]
    )
    ref_U, new_U = (empty_U, full_U) if empty_on_reference else (full_U, empty_U)
    ref_sig, new_sig = (
        (empty_signatures, full_signatures)
        if empty_on_reference
        else (full_signatures, empty_signatures)
    )
    mapping = FeMIFuzzy()._match_centroids(X, ref_U, X, new_U)
    np.testing.assert_array_equal(np.sort(mapping), np.arange(3))
    costs = np.linalg.norm(ref_sig[:, None] - new_sig[None, :], axis=2)
    best = min(
        sum(costs[i, j] for i, j in enumerate(p)) for p in permutations(range(3))
    )
    assert costs[np.arange(3), mapping].sum() == pytest.approx(best)


def test_duplicate_samples_fit_with_two_imputations_and_empty_hard_clusters():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    model = FeMIFuzzy(max_clusters=3, n_imputations=2, random_state=0)
    assert model.fit([X], [["x"]]) is model
    assert model.memberships_.shape == (4, model.n_clusters_)
    assert model.centers_.shape == (model.n_clusters_, 1)
    assert np.isfinite(model.memberships_).all() and np.isfinite(model.centers_).all()
    assert (model.memberships_ >= 0).all()
    np.testing.assert_allclose(model.memberships_.sum(axis=1), 1)
    np.testing.assert_array_equal(model.labels_, model.memberships_.argmax(axis=1))
    np.testing.assert_array_equal(model.predict_proba(), model.memberships_)
    assert model.client_sizes_ == [4]
    assert adjusted_rand_score([0, 0, 1, 1], model.labels_) == 1
