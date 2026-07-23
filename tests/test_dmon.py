"""Unit and integration tests for DMoN (Differentiable Graph Modularity)."""

import pytest
import torch
from soft_clustering import DMoN


@pytest.fixture
def graph_data():
    n, f = 12, 4
    torch.manual_seed(0)
    x = torch.randn(n, f)
    # random edge_index
    edge_index = torch.randint(0, n, (2, 20))
    adj = torch.zeros(n, n)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = torch.max(adj, adj.T)
    adj.fill_diagonal_(0)
    return x, edge_index, adj


def test_forward_shape(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    soft_assign = model(x, edge_index, adj)
    assert soft_assign.shape == (12, 3)


def test_soft_assign_sums_to_one(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    soft_assign = model(x, edge_index, adj)
    sums = soft_assign.sum(dim=1)
    assert torch.allclose(sums, torch.ones(12), atol=1e-5)


def test_loss_scalar(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    soft_assign = model(x, edge_index, adj)
    loss = model.loss(soft_assign, adj)
    assert loss.ndim == 0


def test_k2(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=2)
    soft_assign = model(x, edge_index, adj)
    assert soft_assign.shape == (12, 2)


def test_parameters_exist(graph_data):
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    assert len(list(model.parameters())) > 0
