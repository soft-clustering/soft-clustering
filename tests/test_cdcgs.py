"""Tests for CDCGS (community detection via Gumbel-Softmax).

The estimator is checked against the objective it optimises: on a planted
blockmodel it must reach at least the soft modularity of the planted
partition, and recover that partition.
"""

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from soft_clustering import CDCGS
from soft_clustering._cdcgs import _soft_modularity


def planted_blockmodel(n=60, k=3, p_in=0.6, p_out=0.05, seed=0):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(k), n // k)
    probabilities = np.full((k, k), p_out)
    np.fill_diagonal(probabilities, p_in)
    A = (rng.random((n, n)) < probabilities[z[:, None], z[None, :]]).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A, z


@pytest.fixture
def adj():
    torch.manual_seed(0)
    A = (torch.rand(10, 10) > 0.5).float()
    A = torch.max(A, A.T)
    A.fill_diagonal_(0)
    return A


# ---------------------------------------------------------------------------
# Module behaviour (retained from the pre-estimator API)
# ---------------------------------------------------------------------------


def test_forward_output_shapes(adj):
    model = CDCGS(num_nodes=10, n_clusters=3)
    output, soft_assign = model(adj)
    assert output.shape == (3, 3)
    assert soft_assign.shape == (10, 3)


def test_soft_assign_sums_to_one(adj):
    model = CDCGS(num_nodes=10, n_clusters=3)
    _, soft_assign = model(adj)
    torch.testing.assert_close(soft_assign.sum(dim=1), torch.ones(10), atol=1e-5, rtol=0)


def test_legacy_loss_signature_is_scalar(adj):
    model = CDCGS(num_nodes=10, n_clusters=3, objective="block_diagonal")
    output, _ = model(adj)
    loss = model.loss(output)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_parameters_exist():
    model = CDCGS(num_nodes=10, n_clusters=3)
    assert len(list(model.parameters())) > 0


def test_tau_parameter(adj):
    model = CDCGS(num_nodes=10, n_clusters=3, tau=0.5)
    _, soft_assign = model(adj)
    assert soft_assign.shape == (10, 3)


def test_eval_mode_is_deterministic(adj):
    """A fitted model must not resample Gumbel noise when reading memberships."""
    model = CDCGS(num_nodes=10, n_clusters=3, random_state=0)
    model.eval()
    _, first = model(adj)
    _, second = model(adj)
    torch.testing.assert_close(first, second)


# ---------------------------------------------------------------------------
# Estimator protocol
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    A, _ = planted_blockmodel()
    model = CDCGS(n_clusters=3, random_state=0, max_epochs=30, n_init=1)
    assert model.fit(A) is model


def test_fit_populates_the_protocol_attributes():
    A, _ = planted_blockmodel()
    model = CDCGS(n_clusters=3, random_state=0, max_epochs=30, n_init=1).fit(A)
    U = model.memberships_
    assert U.shape == (A.shape[0], 3)
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(model.labels_, np.argmax(U, axis=1))
    assert model.n_clusters == 3


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        CDCGS(n_clusters=3).predict()


def test_reproducible_under_fixed_seed():
    A, _ = planted_blockmodel()
    kwargs = dict(n_clusters=3, random_state=3, max_epochs=40, n_init=2)
    first = CDCGS(**kwargs).fit(A).memberships_
    second = CDCGS(**kwargs).fit(A).memberships_
    np.testing.assert_allclose(first, second)


def test_infers_node_count_from_the_graph():
    A, _ = planted_blockmodel(n=30, k=3)
    model = CDCGS(n_clusters=3, random_state=0, max_epochs=20, n_init=1).fit(A)
    assert model.num_nodes == 30


# ---------------------------------------------------------------------------
# Optimisation quality
# ---------------------------------------------------------------------------


class TestOptimisation:
    def test_reaches_the_planted_modularity(self):
        A, z = planted_blockmodel(seed=1)
        model = CDCGS(n_clusters=3, random_state=0).fit(A)
        tensor = torch.as_tensor(A)
        planted = float(
            _soft_modularity(torch.as_tensor(np.eye(3, dtype=np.float32)[z]), tensor)
        )
        found = float(
            _soft_modularity(
                torch.as_tensor(model.memberships_.astype(np.float32)), tensor
            )
        )
        assert found >= planted - 1e-3

    def test_recovers_the_planted_partition(self):
        A, z = planted_blockmodel(seed=1)
        model = CDCGS(n_clusters=3, random_state=0).fit(A)
        assert adjusted_rand_score(z, model.labels_) > 0.9

    def test_loss_decreases(self):
        A, _ = planted_blockmodel()
        model = CDCGS(n_clusters=3, random_state=0, n_init=1).fit(A)
        assert model.loss_curve_[-1] < model.loss_curve_[0]

    def test_restarts_do_not_hurt(self):
        """More restarts can only improve the retained objective."""
        A, _ = planted_blockmodel(seed=2)
        one = CDCGS(n_clusters=3, random_state=0, n_init=1).fit(A).best_loss_
        many = CDCGS(n_clusters=3, random_state=0, n_init=4).fit(A).best_loss_
        assert many <= one + 1e-9


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"n_clusters": 0}, {"tau": 0.0}, {"tau_min": 0.0}, {"max_epochs": 0}, {"n_init": 0}],
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        CDCGS(**kwargs)


def test_invalid_objective_raises():
    with pytest.raises(ValueError, match="objective must be"):
        CDCGS(objective="nope")


def test_non_square_input_raises():
    with pytest.raises(ValueError, match="square"):
        CDCGS(n_clusters=2, max_epochs=2).fit(np.zeros((4, 6), dtype=np.float32))


def test_more_clusters_than_nodes_raises():
    A, _ = planted_blockmodel(n=6, k=2)
    with pytest.raises(ValueError, match="smaller than"):
        CDCGS(n_clusters=10, max_epochs=2).fit(A)


def test_empty_graph_is_handled():
    A = np.zeros((12, 12), dtype=np.float32)
    U = CDCGS(n_clusters=3, random_state=0, max_epochs=10, n_init=1).fit(A).memberships_
    assert np.isfinite(U).all()
