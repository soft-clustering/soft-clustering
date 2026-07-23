"""Unit and integration tests for CDCGS (Cluster Detection via Gumbel-Softmax)."""

import pytest
import torch
from soft_clustering import CDCGS


@pytest.fixture
def adj():
    rng = torch.manual_seed(0)
    A = (torch.rand(10, 10) > 0.5).float()
    A = torch.max(A, A.T)
    A.fill_diagonal_(0)
    return A


def test_forward_output_shapes(adj):
    model = CDCGS(num_nodes=10, n_clusters=3)
    output, soft_assign = model(adj)
    assert output.shape == (3, 3)
    assert soft_assign.shape == (10, 3)


def test_soft_assign_sums_to_one(adj):
    model = CDCGS(num_nodes=10, n_clusters=3)
    _, soft_assign = model(adj)
    sums = soft_assign.sum(dim=1)
    assert torch.allclose(sums, torch.ones(10), atol=1e-5)


def test_loss_is_scalar(adj):
    model = CDCGS(num_nodes=10, n_clusters=3)
    output, _ = model(adj)
    loss = model.loss(output)
    assert loss.ndim == 0


def test_loss_nonnegative(adj):
    model = CDCGS(num_nodes=10, n_clusters=3)
    output, _ = model(adj)
    loss = model.loss(output)
    assert loss.item() >= 0


def test_k2(adj):
    model = CDCGS(num_nodes=10, n_clusters=2)
    output, soft_assign = model(adj)
    assert soft_assign.shape == (10, 2)
    assert output.shape == (2, 2)


def test_tau_parameter(adj):
    model = CDCGS(num_nodes=10, n_clusters=3, tau=0.5)
    output, soft_assign = model(adj)
    assert soft_assign.shape == (10, 3)


def test_parameters_exist():
    model = CDCGS(num_nodes=10, n_clusters=3)
    params = list(model.parameters())
    assert len(params) > 0
