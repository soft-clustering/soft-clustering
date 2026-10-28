# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for SoftKSC (soft kernel spectral clustering)."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from soft_clustering import SoftKSC
from soft_clustering._softksc import _soft_memberships


@pytest.fixture
def blobs():
    return make_blobs(150, centers=3, cluster_std=0.7, random_state=0)


def test_recovers_blobs(blobs):
    X, y = blobs
    model = SoftKSC(n_clusters=3, gamma=0.5).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.9


def test_separates_non_convex_clusters():
    X, y = make_moons(200, noise=0.05, random_state=0)
    model = SoftKSC(n_clusters=2, gamma=10.0).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.85


def test_out_of_sample_extension():
    X, _ = make_moons(200, noise=0.05, random_state=0)
    Xt, yt = make_moons(100, noise=0.05, random_state=1)
    model = SoftKSC(n_clusters=2, gamma=10.0).fit(X)
    assert adjusted_rand_score(yt, model.predict(Xt)) > 0.8
    np.testing.assert_allclose(model.predict_proba(X), model.memberships_, atol=1e-10)


def test_dual_problem_is_satisfied(blobs):
    """alpha are eigenvectors of D^-1 M_D Omega (Alzate and Suykens, 2010)."""
    X, _ = blobs
    model = SoftKSC(n_clusters=3, gamma=0.5).fit(X)
    from sklearn.metrics.pairwise import rbf_kernel

    Omega = rbf_kernel(X, gamma=0.5)
    d_inv = 1.0 / Omega.sum(axis=1)
    n = X.shape[0]
    MD = np.eye(n) - np.outer(np.ones(n), d_inv) / d_inv.sum()
    A = d_inv[:, None] * (MD @ Omega)
    for a in model.alphas_.T:
        Aa = A @ a
        lam = (a @ Aa) / (a @ a)
        np.testing.assert_allclose(Aa, lam * a, atol=1e-8)
    # The bias centres the training projections in the D^-1-weighted sense.
    np.testing.assert_allclose(d_inv @ model.projections_, 0.0, atol=1e-8)


def test_membership_formula():
    """Eq. (1.8): sm_q = prod_{j != q} d_j / sum_p prod_{j != p} d_j."""
    E = np.array([[1.0, 0.0]])
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    # cosine distances: 0, 1, 2
    sm = _soft_memberships(E, prototypes)
    expected = np.array([1 * 2, 0 * 2, 0 * 1], dtype=float)
    np.testing.assert_allclose(sm[0], expected / expected.sum())


def test_memberships_are_a_partition(blobs):
    X, _ = blobs
    U = SoftKSC(n_clusters=3, gamma=0.5).fit(X).memberships_
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-12)


def test_legacy_semi_supervised_signature_is_rejected(blobs):
    X, _ = blobs
    with pytest.raises(TypeError):
        SoftKSC().fit(X[:6], np.ones(6), X[6:])


@pytest.mark.parametrize("kwargs", [{"n_clusters": 1}, {"gamma": 0.0}])
def test_invalid_parameters_raise(blobs, kwargs):
    X, _ = blobs
    with pytest.raises(ValueError):
        SoftKSC(**kwargs).fit(X)
