# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for MMSB (Mixed Membership Stochastic Blockmodel).

These check inference, not just shapes: the estimator has to recover a planted
block structure from an observed graph. The previous implementation returned
Dirichlet prior samples that never saw the data, and would fail every test in
``TestRecovery`` below.
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import MMSB


def planted_blockmodel(n=60, k=3, p_in=0.6, p_out=0.05, seed=0):
    """Symmetric graph with `k` equally sized planted blocks."""
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(k), n // k)
    probabilities = np.full((k, k), p_out)
    np.fill_diagonal(probabilities, p_in)
    A = (rng.random((n, n)) < probabilities[z[:, None], z[None, :]]).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A, z


@pytest.fixture
def planted():
    return planted_blockmodel()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_fit_returns_self(planted):
    A, _ = planted
    model = MMSB(n_blocks=3, random_state=0, max_iter=10)
    assert model.fit(A) is model


def test_membership_shape(planted):
    A, _ = planted
    model = MMSB(n_blocks=3, random_state=0, max_iter=10).fit(A)
    assert model.get_memberships().shape == (A.shape[0], 3)
    assert model.memberships_.shape == (A.shape[0], 3)


def test_membership_is_valid_distribution(planted):
    A, _ = planted
    U = MMSB(n_blocks=3, random_state=0, max_iter=10).fit(A).memberships_
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)


def test_block_matrix_shape_and_range(planted):
    A, _ = planted
    B = MMSB(n_blocks=3, random_state=0, max_iter=10).fit(A).get_block_matrix()
    assert B.shape == (3, 3)
    assert np.all(B >= 0) and np.all(B <= 1)


def test_accessors_require_a_fit():
    model = MMSB(n_blocks=3)
    with pytest.raises(RuntimeError):
        model.get_memberships()
    with pytest.raises(RuntimeError):
        model.get_block_matrix()


def test_labels_are_argmax(planted):
    A, _ = planted
    model = MMSB(n_blocks=3, random_state=0, max_iter=10).fit(A)
    np.testing.assert_array_equal(model.labels_, np.argmax(model.memberships_, axis=1))


def test_reproducible_under_fixed_seed(planted):
    A, _ = planted
    first = MMSB(n_blocks=3, random_state=7, max_iter=10).fit(A).memberships_
    second = MMSB(n_blocks=3, random_state=7, max_iter=10).fit(A).memberships_
    np.testing.assert_allclose(first, second)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class TestRecovery:
    """The estimator must learn from the graph it is given."""

    @pytest.mark.parametrize(
        "n,k,p_in,p_out",
        [(60, 3, 0.6, 0.05), (90, 3, 0.4, 0.08), (80, 4, 0.5, 0.05)],
    )
    def test_recovers_planted_partition(self, n, k, p_in, p_out):
        A, z = planted_blockmodel(n=n, k=k, p_in=p_in, p_out=p_out, seed=1)
        model = MMSB(n_blocks=k, random_state=0, max_iter=50).fit(A)
        assert adjusted_rand_score(z, model.labels_) > 0.9

    def test_memberships_depend_on_the_data(self):
        """Two different graphs must not produce the same memberships."""
        first, _ = planted_blockmodel(seed=1)
        second, _ = planted_blockmodel(p_in=0.2, p_out=0.2, seed=2)
        a = MMSB(n_blocks=3, random_state=0, max_iter=20).fit(first).memberships_
        b = MMSB(n_blocks=3, random_state=0, max_iter=20).fit(second).memberships_
        assert not np.allclose(a, b)

    def test_block_matrix_separates_within_from_between(self, planted):
        A, _ = planted
        B = MMSB(n_blocks=3, random_state=0, max_iter=50).fit(A).get_block_matrix()
        off_diagonal = ~np.eye(3, dtype=bool)
        assert np.mean(np.diag(B)) > np.mean(B[off_diagonal])

    def test_spectral_init_beats_the_symmetric_fixed_point(self):
        """Documents why `init='spectral'` is the default.

        Naive mean field started from a uniform partition sits at a fixed
        point where every node is uniformly mixed; the spectral seed is what
        makes the algorithm usable.
        """
        A, z = planted_blockmodel(seed=1)
        spectral = MMSB(n_blocks=3, init="spectral", random_state=0, max_iter=50)
        random = MMSB(n_blocks=3, init="random", random_state=0, max_iter=50)
        spectral_ari = adjusted_rand_score(z, spectral.fit(A).labels_)
        random_ari = adjusted_rand_score(z, random.fit(A).labels_)
        assert spectral_ari > random_ari


# ---------------------------------------------------------------------------
# Generative use
# ---------------------------------------------------------------------------


def test_sample_graph_shape_and_binary():
    Y = MMSB(n_blocks=2, n_nodes=8, random_state=0).sample_graph()
    assert Y.shape == (8, 8)
    assert set(np.unique(Y)).issubset({0.0, 1.0})


def test_sample_graph_needs_a_node_count():
    with pytest.raises(ValueError, match="node count"):
        MMSB(n_blocks=2).sample_graph()


def test_sample_graph_after_fit_uses_the_fitted_size(planted):
    A, _ = planted
    model = MMSB(n_blocks=3, random_state=0, max_iter=5).fit(A)
    assert model.sample_graph().shape == A.shape


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"n_blocks": 0}, {"alpha": 0.0}, {"max_iter": 0}, {"n_inner": 0}]
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        MMSB(**kwargs)


def test_invalid_init_raises():
    with pytest.raises(ValueError, match="init must be"):
        MMSB(init="nope")


def test_non_square_input_raises():
    with pytest.raises(ValueError, match="square"):
        MMSB(n_blocks=2, max_iter=2).fit(np.zeros((4, 6)))


def test_more_blocks_than_nodes_raises():
    A, _ = planted_blockmodel(n=6, k=2)
    with pytest.raises(ValueError, match="smaller than"):
        MMSB(n_blocks=10, max_iter=2).fit(A)


def test_empty_graph_is_handled():
    """A graph with no edges must still produce a valid partition."""
    A = np.zeros((12, 12))
    U = MMSB(n_blocks=3, random_state=0, max_iter=5).fit(A).memberships_
    assert np.isfinite(U).all()
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)
