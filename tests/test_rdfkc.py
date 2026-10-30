# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for RDFKC (robust deep fuzzy k-means clustering)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="RDFKC needs the `deep` extra")

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from soft_clustering import RDFKC  # noqa: E402
from soft_clustering._rdfkc import _simplex_quadratic  # noqa: E402


def _autoencoder(d=8, latent=3):
    nn = torch.nn
    encoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, latent))
    decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, d))
    return encoder, decoder


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.4, (30, 8)) for c in (-2, 0, 2)]).astype("float32")
    return X, np.repeat([0, 1, 2], 30)


def test_recovers_clusters_with_soft_memberships(blobs):
    """Regression: memberships_ used to be the one-hot of the hard labels."""
    X, y = blobs
    encoder, decoder = _autoencoder()
    model = RDFKC(
        n_clusters=3,
        encoder=encoder,
        decoder=decoder,
        random_state=0,
        max_iter=40,
        lr=1e-3,
    )
    U = model.fit_predict(X)
    assert adjusted_rand_score(y, model.labels_) > 0.9
    assert np.unique(U.round(4)).size > 3, "memberships are not soft"
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-5)


def test_simplex_solver_matches_the_kkt_conditions():
    rng = np.random.default_rng(1)
    p = torch.tensor(rng.normal(size=(50, 4)), dtype=torch.float64)
    q = torch.tensor(rng.uniform(0.1, 2.0, size=(50, 4)), dtype=torch.float64)
    U = _simplex_quadratic(p, q).numpy()
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-12)
    # Brute force on a grid for one row: the solver attains the minimum.
    from scipy.optimize import minimize

    for i in range(3):

        def f(u, i=i):
            return float(np.sum(q[i].numpy() * u**2 - p[i].numpy() * u))

        ref = minimize(
            f,
            np.full(4, 0.25),
            bounds=[(0, 1)] * 4,
            constraints={"type": "eq", "fun": lambda u: u.sum() - 1},
        )
        assert f(U[i]) <= ref.fun + 1e-8


def test_reproducible_and_leaves_global_rng_alone(blobs):
    X, _ = blobs
    encoder, decoder = _autoencoder()
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    a = RDFKC(
        n_clusters=3, encoder=encoder, decoder=decoder, random_state=4, max_iter=3
    ).fit(X)
    assert torch.rand(1) == expected_next
    b = RDFKC(
        n_clusters=3, encoder=encoder, decoder=decoder, random_state=4, max_iter=3
    ).fit(X)
    np.testing.assert_allclose(a.memberships_, b.memberships_)


def test_caller_modules_are_not_trained_in_place(blobs):
    X, _ = blobs
    encoder, decoder = _autoencoder()
    before = [w.detach().clone() for w in encoder.parameters()]
    RDFKC(
        n_clusters=3, encoder=encoder, decoder=decoder, random_state=0, max_iter=2
    ).fit(X)
    for w, w0 in zip(encoder.parameters(), before):
        assert torch.equal(w.detach(), w0)


def test_bundled_fashion_architecture_runs():
    rng = np.random.default_rng(38)
    X = rng.random((8, 1, 28, 28)).astype(np.float32)
    model = RDFKC(n_clusters=2, dataset="fashion", random_state=0, max_iter=1)
    assert model.fit_predict(X).shape == (8, 2)


def test_missing_networks_raise():
    with pytest.raises(ValueError, match="encoder and decoder"):
        RDFKC(n_clusters=3)


@pytest.mark.parametrize("n", [2, 4])
def test_equal_codes_use_every_other_sample(n):
    model = RDFKC(
        n_clusters=2, encoder=torch.nn.Identity(), decoder=torch.nn.Identity()
    )
    actual = model._similarity(torch.zeros(n, 2))
    torch.testing.assert_close(actual, torch.ones(n, n) - torch.eye(n))


@pytest.mark.parametrize("values", [[0.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
def test_nonself_heat_weights_for_mixed_and_distinct_codes(values):
    model = RDFKC(
        n_clusters=2, encoder=torch.nn.Identity(), decoder=torch.nn.Identity()
    )
    Z = torch.tensor(values)[:, None]
    distances = np.abs(np.subtract.outer(values, values))
    expected = np.exp(-distances / 0.5)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(
        model._similarity(Z).numpy(), expected, rtol=1e-6, atol=1e-8
    )


def test_tied_neighbor_cutoff_preserves_total_graph_weight():
    model = RDFKC(
        n_clusters=2, encoder=torch.nn.Identity(), decoder=torch.nn.Identity()
    )
    actual = model._similarity(torch.zeros(13, 2), k=10)
    torch.testing.assert_close(actual.diag(), torch.zeros(13))
    torch.testing.assert_close(actual, actual.T)
    assert actual.sum().item() == pytest.approx(130.0)


@pytest.mark.parametrize("n,k", [(1, 10), (4, 0)])
def test_private_zero_neighbor_graph_is_unchanged(n, k):
    model = RDFKC(
        n_clusters=2, encoder=torch.nn.Identity(), decoder=torch.nn.Identity()
    )
    torch.testing.assert_close(
        model._similarity(torch.zeros(n, 2), k=k), torch.zeros(n, n)
    )


def test_one_round_equal_codes_average_other_initial_memberships():
    X = np.zeros((4, 2), dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        encoder = torch.nn.Linear(2, 2, bias=False)
        decoder = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            encoder.weight.copy_(torch.eye(2))
            decoder.weight.copy_(torch.eye(2))
    kwargs = dict(
        n_clusters=2,
        encoder=encoder,
        decoder=decoder,
        random_state=0,
        mu=1.0,
        gamma=0.0,
        tau=0.1,
    )
    initial = RDFKC(max_iter=0, **kwargs).fit(X)
    model = RDFKC(max_iter=1, **kwargs).fit(X)
    expected = np.array(
        [np.delete(initial.memberships_, i, axis=0).mean(axis=0) for i in range(len(X))]
    )
    np.testing.assert_array_equal(model.latent_, np.zeros((4, 2)))
    np.testing.assert_allclose(model.memberships_, expected, rtol=1e-6, atol=1e-6)


class _ConstantEncoder(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, X):
        return self.value.expand(len(X), 1)


class _ConstantDecoder(torch.nn.Module):
    def __init__(self, sample_shape):
        super().__init__()
        self.sample_shape = sample_shape
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, Z):
        return self.value.expand(len(Z), *self.sample_shape)


@pytest.fixture
def recorded_gradients(monkeypatch):
    """Observe the production backward pass without changing the parameters."""
    snapshots = []

    class RecordingOptimizer:
        def __init__(self, params, lr):
            self.params = list(params)

        def zero_grad(self):
            for param in self.params:
                param.grad = None

        def step(self):
            snapshots.append([param.grad.detach().clone() for param in self.params])

    monkeypatch.setattr(torch.optim, "Adam", RecordingOptimizer)
    return snapshots


def test_fit_network_gradient_holds_robust_weights_fixed(recorded_gradients):
    # These modules have no reset_parameters or random forward operations.
    # Replay only U/V initialization to put both samples exactly one unit from V.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        torch.rand(2, 1)
        center = torch.randn(1, 1).item()
    RDFKC(
        n_clusters=1,
        encoder=_ConstantEncoder(center + 1.0),
        decoder=_ConstantDecoder((1,)),
        random_state=0,
        max_iter=1,
        gamma=0.0,
        tau=0.1,
    ).fit(np.zeros((2, 1), dtype=np.float32))
    # Two samples, U=1, d=1: sum ell'(d), with no reconstruction path to z.
    expected = 2 * (1.0 + 0.1) * (1.0 + 2 * 0.1) / (1.0 + 0.1) ** 2
    assert len(recorded_gradients) == 1
    assert recorded_gradients[0][0].item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "sample_values,sample_shape", [([0.0, 1.0, 2.0], (8,)), ([1.0, 1.0], (1, 28, 28))]
)
def test_fit_reconstruction_averages_samples_not_elements(
    recorded_gradients, sample_values, sample_shape
):
    X = np.ones((len(sample_values), *sample_shape), dtype=np.float32)
    X *= np.array(sample_values, dtype=np.float32).reshape(
        len(sample_values), *([1] * len(sample_shape))
    )
    RDFKC(
        n_clusters=1,
        encoder=_ConstantEncoder(1.0),
        decoder=_ConstantDecoder(sample_shape),
        random_state=0,
        max_iter=1,
        batch_size=2,
        gamma=0.0,
    ).fit(X)
    # Clustering cannot reach the decoder's shared scalar. The unequal final
    # batch has a different mean in the feature case: an unweighted mean of
    # batch means would give gradient -20 instead of -16. Both sample means are 1.
    assert len(recorded_gradients) == 1
    assert recorded_gradients[0][1].item() == pytest.approx(-2 * np.prod(sample_shape))


def _batched_latent(model, X):
    with torch.no_grad():
        return torch.cat(
            [model.encoder_(part) for part in torch.as_tensor(X).split(2)]
        ).numpy()


def test_one_round_state_uses_current_encoder_for_prototypes_and_memberships():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]], dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder, decoder = _autoencoder(d=2, latent=2)
    kwargs = dict(
        n_clusters=2,
        encoder=encoder,
        decoder=decoder,
        random_state=0,
        batch_size=2,
        lr=0.1,
        mu=0.0,
    )
    initial = RDFKC(max_iter=0, **kwargs).fit(X)
    model = RDFKC(max_iter=1, **kwargs).fit(X)
    Z = _batched_latent(model, X).astype(np.float64)
    assert not np.allclose(Z, initial.latent_)
    np.testing.assert_array_equal(model.latent_, Z)
    # Independent NumPy oracle for the single V/U round, with graph term zero.
    d0 = np.linalg.norm(Z[:, None] - initial.latent_centers_[None], axis=2)
    k0 = (1 + model.tau) * (d0 + 2 * model.tau) / (2 * (d0 + model.tau) ** 2)
    w0 = initial.memberships_**2 * k0
    expected_V = (w0.T @ Z) / (w0.sum(0)[:, None] + 1e-8)
    d1 = np.linalg.norm(Z[:, None] - expected_V[None], axis=2) + 1e-8
    q1 = (1 + model.tau) * d1**2 / (d1 + model.tau)
    assert q1.min() > 1e-12
    expected_U = 1 / q1
    expected_U /= expected_U.sum(1, keepdims=True)
    np.testing.assert_allclose(model.latent_centers_, expected_V, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.memberships_, expected_U, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("rounds", [0, 1])
def test_eval_refresh_preserves_modes_buffers_and_batching(rounds):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]], dtype=np.float32)
    nn = torch.nn
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(0.5))
        decoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(0.25))
    decoder.eval()  # Preserve even different caller modes, not just parameters.
    originals = (encoder, decoder)
    modes = [[module.training for module in net.modules()] for net in originals]
    states = [
        {key: value.clone() for key, value in net.state_dict().items()}
        for net in originals
    ]
    rng_before = torch.random.get_rng_state().clone()
    model = RDFKC(
        n_clusters=2,
        encoder=encoder,
        decoder=decoder,
        random_state=0,
        max_iter=rounds,
        batch_size=2,
    ).fit(X)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for trained in (model.encoder_, model.decoder_):
        assert all(not module.training for module in trained.modules())
        assert trained[1].num_batches_tracked.item() == 2 * rounds
    np.testing.assert_array_equal(model.latent_, _batched_latent(model, X))
    for net, saved_modes, saved_state in zip(originals, modes, states):
        assert [module.training for module in net.modules()] == saved_modes
        for key, value in net.state_dict().items():
            assert torch.equal(value, saved_state[key])
