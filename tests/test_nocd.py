# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for NOCD (Network Overlapping Community Detection)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the `deep` extra")

import scipy.sparse as sp  # noqa: E402

from soft_clustering import NOCD, _nocd  # noqa: E402


@pytest.mark.parametrize("n,seed,zero_rows", [(500, 5, True), (1000, 0, False)])
@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_full_loss_and_gradients_match_explicit_pairs(
    n, seed, zero_rows, balanced, dtype, device
):
    """The oracle enumerates nonedges; it never subtracts large pair totals."""
    A = np.ones((n, n)) - np.eye(n)
    A[0, 1] = A[1, 0] = 0
    adj = sp.csr_matrix(A)
    values = np.random.default_rng(seed).uniform(0.1, 1.0, (n, 2)).astype(np.float32)
    if zero_rows:
        values[:2] = 0.0
    emb = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
    loss = _nocd._BerpoDecoder(n, adj.nnz, balance_loss=balanced).loss_full(emb, adj)
    loss.backward()

    reference = torch.tensor(
        values, dtype=torch.float64, device=device, requires_grad=True
    )
    edges = torch.tensor(np.transpose(adj.nonzero()), dtype=torch.long, device=device)
    nonedge_mask = A == 0
    np.fill_diagonal(nonedge_mask, False)
    nonedges = torch.tensor(np.argwhere(nonedge_mask), dtype=torch.long, device=device)
    assert nonedges.shape == (2, 2)
    eps = -np.log(1.0 - len(edges) / (n * (n - 1)))
    edge_dots = (reference[edges[:, 0]] * reference[edges[:, 1]]).sum(dim=1)
    positive = -torch.log(-torch.expm1(-(eps + edge_dots))).mean()
    negative = (reference[nonedges[:, 0]] * reference[nonedges[:, 1]]).sum(dim=1).mean()
    scale = 1.0 if balanced else len(nonedges) / len(edges)
    expected = (positive + scale * negative) / (1.0 + scale)
    expected.backward()

    assert loss.dtype == emb.dtype == emb.grad.dtype
    assert loss.device == emb.device == emb.grad.device
    assert torch.isfinite(loss) and torch.isfinite(emb.grad).all()
    torch.testing.assert_close(loss.double(), expected, rtol=2e-6, atol=1e-10)
    # Check small edge-only gradients separately from the missing-edge endpoints.
    for rows in (slice(0, 2), slice(2, None)):
        torch.testing.assert_close(
            emb.grad[rows].double(), reference.grad[rows], rtol=2e-5, atol=1e-10
        )


@pytest.fixture
def graph():
    rng = np.random.default_rng(37)
    n = 20
    A = (rng.random((n, n)) > 0.6).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    adj = sp.csr_matrix(A)
    features = sp.eye(n, format="csr")
    return adj, features, n


def test_fit_predict_runs(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16], batch_size=64)
    memberships = model.fit_predict(adj, features, K=3)
    assert memberships is not None


def test_memberships_shape(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16], batch_size=64)
    memberships = model.fit_predict(adj, features, K=3)
    assert memberships.shape == (n, 3)


def test_memberships_nonneg(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16], batch_size=64)
    memberships = model.fit_predict(adj, features, K=3)
    assert np.all(memberships >= 0)


def test_k2(graph):
    adj, features, n = graph
    model = NOCD(random_state=0, max_epochs=5, hidden_sizes=[16], batch_size=64)
    memberships = model.fit_predict(adj, features, K=2)
    assert memberships.shape == (n, 2)


def test_sampler_and_dense_graph_fits_terminate_in_bounded_child():
    """A regression must time out in a child instead of hanging the test runner."""
    code = textwrap.dedent("""
        import numpy as np
        import scipy.sparse as sp
        import torch
        from soft_clustering import NOCD, _nocd

        torch.set_num_threads(1)
        torch.cuda.is_available = lambda: False
        complete = np.ones((10, 10)) - np.eye(10)
        try:
            _nocd._EdgeSampler(sp.csr_matrix(complete), num_pos=8, num_neg=8)[0]
        except ValueError as exc:
            assert "edge and one off-diagonal nonedge" in str(exc)
        else:
            raise AssertionError("complete graph accepted by sampler")
        for A in (complete, np.zeros((10, 10)), np.zeros((1, 1))):
            for stochastic in (False, True):
                for balanced in (False, True):
                    try:
                        NOCD(n_clusters=1, max_epochs=1, stochastic_loss=stochastic,
                             balance_loss=balanced).fit(A)
                    except ValueError as exc:
                        assert "edge and one off-diagonal nonedge" in str(exc)
                    else:
                        raise AssertionError("unsupported graph accepted")
        A = complete.copy()
        A[0, 1] = A[1, 0] = 0
        sampler = _nocd._EdgeSampler(sp.csr_matrix(A), 64, 64, base_seed=7)
        positive, negative = sampler[3]
        assert positive.shape == negative.shape == (64, 2)
        assert np.all(A[positive[:, 0], positive[:, 1]] != 0)
        assert np.all(positive[:, 0].numpy() != positive[:, 1].numpy())
        assert set(map(tuple, negative.tolist())) == {(0, 1), (1, 0)}
        for actual, repeated in zip((positive, negative), sampler[3]):
            assert torch.equal(actual, repeated)
        params = dict(n_clusters=2, max_epochs=1, hidden_sizes=[4], dropout=0.0,
                      batch_norm=False, batch_size=64, random_state=0)
        for stochastic in (True, False):
            if not stochastic:
                def forbidden(*args, **kwargs):
                    raise AssertionError("full-loss mode constructed a sampler")
                _nocd._get_edge_sampler = forbidden
            model = NOCD(stochastic_loss=stochastic, **params).fit(A)
            assert np.isfinite(model.memberships_).all()
            assert np.isfinite(model.affiliations_).all()
        print("complete/empty graphs rejected; near-complete sampling and both fits completed")
    """)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1])
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "both fits completed" in result.stdout


def test_rank_sampler_covers_exact_weighted_nonedge_set_without_mutation(monkeypatch):
    A = sp.csr_matrix(
        (
            np.array([9.0, 2.0, 3.0, 0.0, 0.5, 7.0, 1.5, 4.0, 3.0, 4.0, 2.0]),
            np.array([0, 1, 1, 2, 0, 1, 2, 2, 3, 0, 3]),
            np.array([0, 4, 7, 9, 11]),
        ),
        shape=(4, 4),
    )
    saved = [part.copy() for part in (A.data, A.indices, A.indptr)]
    expected_edges = {(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)}
    allowed = {
        (u, v)
        for u in range(4)
        for v in range(4)
        if u != v and (u, v) not in expected_edges
    }
    sampler = _nocd._EdgeSampler(A, num_pos=5, num_neg=len(allowed))

    class AllRanks:
        def integers(self, low, high, size, dtype):
            assert low == 0 and size == high
            return np.arange(high, dtype=dtype)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: AllRanks())
    edges, nonedges = sampler[0]
    assert set(map(tuple, edges.tolist())) == expected_edges
    assert set(map(tuple, nonedges.tolist())) == allowed
    assert sampler.A[0, 1] == 5 and sampler.A[1, 0] == 0.5
    assert np.all(sampler.A.diagonal() == 0)
    assert sampler.A.nnz == 5
    for actual, original in zip((A.data, A.indices, A.indptr), saved):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("balanced", [False, True])
def test_decoder_full_and_exhaustive_batch_losses_agree(balanced):
    A = _nocd._prepare_adjacency(
        sp.csr_matrix([[9.0, 2.0, 0.0], [0.5, 0.0, 0.0], [0.0, 3.0, 4.0]])
    )
    positive = torch.tensor(np.transpose(A.nonzero()), dtype=torch.long)
    negative = torch.tensor(
        [(u, v) for u in range(3) for v in range(3) if u != v and A[u, v] == 0]
    )
    emb = torch.tensor([[0.2, 0.4], [0.5, 0.3], [0.8, 0.6]])
    decoder = _nocd._BerpoDecoder(3, A.nnz, balance_loss=balanced)
    torch.testing.assert_close(
        decoder.loss_full(emb, A), decoder.loss_batch(emb, positive, negative)
    )
    for n, m in ((0, 0), (1, 0), (10, 0), (10, 90), (10, 91), (10, -1)):
        with np.errstate(all="raise"), pytest.raises(ValueError, match="off-diagonal"):
            _nocd._BerpoDecoder(n, m, balance_loss=balanced)


@pytest.mark.parametrize(
    "budget,scores,checkpoints,best,patience",
    [
        (0, [1.0], [0], 0, 10),
        (1, [5.0, 1.0], [0, 1], 1, 10),
        (25, [1.0, 5.0], [0, 25], 0, 10),
        (26, [5.0, 1.0, 4.0], [0, 25, 26], 1, 10),
        (26, [5.0, 1.0, np.nan], [0, 25, 26], 1, 10),
        (100, [1.0, 2.0, 3.0], [0, 25, 50], 0, 1),
    ],
)
def test_update_budget_and_best_trained_checkpoint(
    monkeypatch, budget, scores, checkpoints, best, patience
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    steps, sampled, candidates, validated = [], [], [], []
    original_step = torch.optim.Adam.step
    original_getitem = _nocd._EdgeSampler.__getitem__
    original_sampler = _nocd._get_edge_sampler
    original_monitor = _nocd._NoImprovementStopping

    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        steps.append(1)
        return result

    def sample(self, key):
        sampled.append(key)
        return original_getitem(self, key)

    def make_sampler(*args, **kwargs):
        assert budget > 0, "zero budget must not construct a sampler"
        return original_sampler(*args, **kwargs)

    def validation(self, emb, adj):
        assert not torch.is_grad_enabled()
        validated.append(len(steps))
        candidates.append(emb.detach().cpu().numpy().copy())
        return torch.tensor(scores[len(candidates) - 1])

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(_nocd._EdgeSampler, "__getitem__", sample)
    monkeypatch.setattr(_nocd, "_get_edge_sampler", make_sampler)
    monkeypatch.setattr(_nocd._BerpoDecoder, "loss_full", validation)
    monkeypatch.setattr(
        _nocd,
        "_NoImprovementStopping",
        lambda fn, **kwargs: original_monitor(fn, patience=patience),
    )
    A = np.roll(np.eye(12), 1, axis=1) + np.roll(np.eye(12), -1, axis=1)
    model = NOCD(
        n_clusters=2,
        random_state=0,
        max_epochs=budget,
        hidden_sizes=[4],
        dropout=0.0,
        batch_norm=False,
        batch_size=8,
        lr=0.01,
    ).fit(A)
    assert validated == checkpoints
    assert len(steps) == len(sampled) == checkpoints[-1]
    np.testing.assert_allclose(model.affiliations_, candidates[best])
    top = model.affiliations_.max()
    np.testing.assert_allclose(
        model.memberships_, model.affiliations_ / top if top else model.affiliations_
    )


def test_negative_update_budget_rejects(graph):
    with pytest.raises(ValueError, match="max_epochs must be nonnegative"):
        NOCD(max_epochs=-1).fit(graph[0])
