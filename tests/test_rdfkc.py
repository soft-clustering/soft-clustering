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
