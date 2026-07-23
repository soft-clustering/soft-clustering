"""Unit and integration tests for MMSB (Mixed Membership Stochastic Blockmodel)."""

import pytest
import torch
from soft_clustering import MMSB


def test_membership_shape():
    model = MMSB(n_nodes=10, n_blocks=3)
    pi = model.get_memberships()
    assert pi.shape == (10, 3)


def test_membership_is_valid_distribution():
    model = MMSB(n_nodes=10, n_blocks=3)
    pi = model.get_memberships()
    assert torch.all(pi >= 0)
    # Rows drawn from Dirichlet — sum to ~1
    row_sums = pi.sum(dim=1)
    for s in row_sums:
        assert abs(s.item() - 1.0) < 1e-4


def test_block_matrix_shape():
    model = MMSB(n_nodes=10, n_blocks=3)
    B = model.get_block_matrix()
    assert B.shape == (3, 3)


def test_block_matrix_in_range():
    model = MMSB(n_nodes=10, n_blocks=3)
    B = model.get_block_matrix()
    assert torch.all(B >= 0) and torch.all(B <= 1)


def test_sample_graph_shape():
    model = MMSB(n_nodes=8, n_blocks=2)
    Y = model.sample_graph()
    assert Y.shape == (8, 8)


def test_sample_graph_binary():
    model = MMSB(n_nodes=8, n_blocks=2)
    Y = model.sample_graph()
    assert set(Y.unique().tolist()).issubset({0.0, 1.0})


def test_different_alpha():
    model = MMSB(n_nodes=10, n_blocks=3, alpha=1.0)
    pi = model.get_memberships()
    assert pi.shape == (10, 3)
