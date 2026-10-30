# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for DMoN (Deep Modularity Networks).

The encoder is checked against the objective it optimises and against a
planted community structure, not only for output shape.
"""

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the `deep` extra")

import torch  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from soft_clustering import DMoN  # noqa: E402


def planted_graph(n=60, k=3, p_in=0.6, p_out=0.05, n_features=4, seed=0):
    """Planted blockmodel with block-informative node features."""
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(k), n // k)
    probabilities = np.full((k, k), p_out)
    np.fill_diagonal(probabilities, p_in)
    adj = (rng.random((n, n)) < probabilities[z[:, None], z[None, :]]).astype(
        np.float32
    )
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)

    x = np.zeros((n, n_features), dtype=np.float32)
    x[np.arange(n), z % n_features] = 1.0
    x += rng.normal(0, 0.3, x.shape).astype(np.float32)
    return x, adj, z


@pytest.fixture
def graph_data():
    n, f = 12, 4
    torch.manual_seed(0)
    x = torch.randn(n, f)
    edge_index = torch.randint(0, n, (2, 20))
    adj = torch.zeros(n, n)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = torch.max(adj, adj.T)
    adj.fill_diagonal_(0)
    return x, edge_index, adj


# ---------------------------------------------------------------------------
# Module behaviour
# ---------------------------------------------------------------------------


def test_forward_shape(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    assert model(x, edge_index, adj).shape == (12, 3)


def test_soft_assign_sums_to_one(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    soft_assign = model(x, edge_index, adj)
    torch.testing.assert_close(
        soft_assign.sum(dim=1), torch.ones(12), atol=1e-5, rtol=0
    )


def test_loss_scalar(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    loss = model.loss(model(x, edge_index, adj), adj)
    assert loss.ndim == 0


def test_k2(graph_data):
    x, edge_index, adj = graph_data
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=2)
    assert model(x, edge_index, adj).shape == (12, 2)


def test_parameters_exist():
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3)
    assert len(list(model.parameters())) > 0


def test_collapse_regulariser_penalises_a_single_cluster():
    """A degenerate assignment must cost more than a balanced one."""
    model = DMoN(in_channels=2, hidden_channels=4, n_clusters=3)
    adj = torch.ones(9, 9) - torch.eye(9)

    collapsed = torch.zeros(9, 3)
    collapsed[:, 0] = 1.0
    balanced = torch.zeros(9, 3)
    balanced[torch.arange(9), torch.arange(9) % 3] = 1.0

    assert model.loss(collapsed, adj) > model.loss(balanced, adj)


# ---------------------------------------------------------------------------
# Estimator protocol
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    x, adj, _ = planted_graph()
    model = DMoN(in_channels=4, hidden_channels=8, n_clusters=3, max_epochs=20)
    assert model.fit(x, adj=adj) is model


def test_fit_populates_the_protocol_attributes():
    x, adj, _ = planted_graph()
    model = DMoN(
        in_channels=4, hidden_channels=8, n_clusters=3, random_state=0, max_epochs=30
    ).fit(x, adj=adj)
    U = model.memberships_
    assert U.shape == (x.shape[0], 3)
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(model.labels_, np.argmax(U, axis=1))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DMoN(in_channels=4, hidden_channels=8, n_clusters=3).predict()


def test_reproducible_under_fixed_seed():
    x, adj, _ = planted_graph()
    kwargs = dict(
        in_channels=4, hidden_channels=8, n_clusters=3, random_state=5, max_epochs=25
    )
    first = DMoN(**kwargs).fit(x, adj=adj).memberships_
    second = DMoN(**kwargs).fit(x, adj=adj).memberships_
    np.testing.assert_allclose(first, second)


def test_edge_index_and_adj_paths_agree():
    x, adj, _ = planted_graph()
    edge_index = torch.as_tensor(np.array(np.nonzero(adj)))
    kwargs = dict(
        in_channels=4, hidden_channels=8, n_clusters=3, random_state=0, max_epochs=25
    )
    from_adj = DMoN(**kwargs).fit(x, adj=adj).memberships_
    from_edges = DMoN(**kwargs).fit(x, edge_index=edge_index).memberships_
    np.testing.assert_allclose(from_adj, from_edges, atol=1e-5)


# ---------------------------------------------------------------------------
# Optimisation quality
# ---------------------------------------------------------------------------


def _fit_with_loss_schedule(model, x, adj, losses, monkeypatch):
    """Control monitored values while retaining real gradients and Adam steps."""
    original_loss = model.loss
    targets = iter(losses)
    original_adam = torch.optim.Adam
    updates = 0
    handles = []

    def scheduled_loss(soft_assign, adjacency):
        real_loss = original_loss(soft_assign, adjacency)
        return real_loss - real_loss.detach() + real_loss.new_tensor(next(targets))

    def count_update(optimizer, args, kwargs):
        nonlocal updates
        updates += 1

    def observed_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        handles.append(optimizer.register_step_post_hook(count_update))
        return optimizer

    try:
        with monkeypatch.context() as patch:
            patch.setattr(model, "loss", scheduled_loss)
            patch.setattr(torch.optim, "Adam", observed_adam)
            model.fit(x, adj=adj)
    finally:
        for handle in handles:
            handle.remove()
    return updates


@pytest.mark.parametrize(
    "losses,tol,budget,expected",
    [
        pytest.param([10, 10, 10, 10], 1.0, 4, 3, id="confirmed-plateau"),
        pytest.param(
            [10, 8, 7.75, 6.75, 6.5, 6.25, 5],
            1.0,
            7,
            6,
            id="reset-at-strict-boundary",
        ),
        pytest.param(
            [10, 9.5, 11, 10.5, 9, 9.25, 9.5, 8],
            1.0,
            8,
            7,
            id="large-changes-reset-small-rises-qualify",
        ),
        pytest.param([10, 8, 6, 4], 1.0, 4, 4, id="material-progress"),
        pytest.param([10] * 5, 0.0, 5, 5, id="zero-tolerance"),
        pytest.param([10], 1.0, 1, 1, id="one-update-budget"),
        pytest.param([10, 10], 1.0, 2, 2, id="two-update-budget"),
    ],
)
def test_stopping_requires_consecutive_small_changes(
    losses, tol, budget, expected, monkeypatch
):
    x, adj, _ = planted_graph(n=12)
    model = DMoN(
        in_channels=4, hidden_channels=8, random_state=0, max_epochs=budget, tol=tol
    )
    assert _fit_with_loss_schedule(model, x, adj, losses, monkeypatch) == expected
    assert model.loss_curve_ == losses[:expected]


def test_stopping_publishes_updated_state_and_resets_on_refit(monkeypatch):
    x, adj, _ = planted_graph(n=12)
    kwargs = dict(in_channels=4, hidden_channels=8, random_state=0)
    model = DMoN(**kwargs, max_epochs=4, tol=1.0)
    losses = [10, 10, 10, 10]
    assert _fit_with_loss_schedule(model, x, adj, losses, monkeypatch) == 3
    assert model.loss_curve_ == losses[:3]

    control = DMoN(**kwargs, max_epochs=3, tol=0.0).fit(x, adj=adj)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[name])
    np.testing.assert_allclose(model.memberships_, control.memberships_)

    model.eval()
    adjacency = torch.as_tensor(adj)
    edges = adjacency.nonzero(as_tuple=False).T.contiguous()
    with torch.no_grad():
        final_memberships = model(torch.as_tensor(x), edges, adjacency).numpy()
    np.testing.assert_allclose(model.memberships_, final_memberships)

    assert _fit_with_loss_schedule(model, x, adj, losses, monkeypatch) == 3
    assert model.loss_curve_ == losses[:3]
    np.testing.assert_allclose(model.memberships_, control.memberships_)


def test_recovers_planted_communities():
    x, adj, z = planted_graph(seed=1)
    model = DMoN(
        in_channels=4, hidden_channels=16, n_clusters=3, random_state=0, max_epochs=300
    ).fit(x, adj=adj)
    assert adjusted_rand_score(z, model.labels_) > 0.9


def test_loss_decreases():
    x, adj, _ = planted_graph()
    model = DMoN(
        in_channels=4, hidden_channels=16, n_clusters=3, random_state=0, max_epochs=100
    ).fit(x, adj=adj)
    assert model.loss_curve_[-1] < model.loss_curve_[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"n_clusters": 0}, {"max_epochs": 0}])
def test_invalid_hyperparameters_raise(kwargs):
    base = dict(in_channels=4, hidden_channels=8, n_clusters=3)
    with pytest.raises(ValueError):
        DMoN(**{**base, **kwargs})


def test_fit_needs_a_graph():
    x, _, _ = planted_graph()
    with pytest.raises(ValueError, match="edge_index"):
        DMoN(in_channels=4, hidden_channels=8, n_clusters=3).fit(x)


def test_more_clusters_than_nodes_raises():
    x, adj, _ = planted_graph(n=6, k=2)
    with pytest.raises(ValueError, match="smaller than"):
        DMoN(in_channels=4, hidden_channels=8, n_clusters=10, max_epochs=2).fit(
            x, adj=adj
        )


def test_empty_graph_is_handled():
    x, adj, _ = planted_graph()
    adj = np.zeros_like(adj)
    U = (
        DMoN(
            in_channels=4,
            hidden_channels=8,
            n_clusters=3,
            random_state=0,
            max_epochs=10,
        )
        .fit(x, adj=adj)
        .memberships_
    )
    assert np.isfinite(U).all()


def test_gcn_layer_matches_torch_geometric():
    """DMoN's built-in graph convolution reproduces torch_geometric's GCNConv.

    torch_geometric used to be a dependency for this one layer. The check runs
    only where torch_geometric happens to be installed.
    """
    import torch

    gcn = pytest.importorskip("torch_geometric.nn", reason="reference not installed")
    from soft_clustering._dmon import _GCNLayer

    torch.manual_seed(0)
    n = 12
    edge_index = torch.randint(0, n, (2, 30))
    edge_index = torch.cat([edge_index, torch.tensor([[3], [3]])], dim=1)
    x = torch.randn(n, 5)
    reference, ours = gcn.GCNConv(5, 4), _GCNLayer(5, 4)
    with torch.no_grad():
        ours.weight.copy_(reference.lin.weight.T)
        reference.bias.copy_(torch.randn(4))
        ours.bias.copy_(reference.bias)
    propagation = _GCNLayer.propagation_matrix(edge_index, n)
    assert torch.allclose(reference(x, edge_index), ours(x, propagation), atol=1e-6)
