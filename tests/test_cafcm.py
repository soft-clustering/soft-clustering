# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for CAFCM (collaborative annealing fuzzy c-means)."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import CAFCM


@pytest.fixture
def blobs():
    rng = np.random.default_rng(5)
    X = np.vstack(
        [rng.normal(c, 0.5, (40, 2)) for c in ([0, 0], [4, 0], [0, 4], [4, 4])]
    )
    return X, np.repeat(np.arange(4), 40)


def test_recovers_planted_clusters(blobs):
    X, y = blobs
    model = CAFCM(n_clusters=4, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.95


def test_schedule_anneals_towards_crisp(blobs):
    X, _ = blobs
    model = CAFCM(
        n_clusters=4, m_start=2.0, m_end=1.05, cooling_rate=0.7, random_state=0
    )
    model.fit(X)
    schedule = model.fuzzifier_schedule_
    assert schedule[0] == pytest.approx(2.0)
    assert schedule[-1] == pytest.approx(1.05)
    assert np.all(np.diff(schedule) < 0)
    # At m close to one the memberships are nearly crisp.
    assert model.memberships_.max(axis=1).mean() > 0.95


def test_one_best_module_is_recorded_per_temperature(blobs):
    X, _ = blobs
    model = CAFCM(n_clusters=4, n_modules=3, random_state=1).fit(X)
    assert model.sse_trajectory_.shape == model.fuzzifier_schedule_.shape
    # The reported partition is the best module's at the final temperature.
    d2 = ((X[:, None, :] - model.centers_[None, :, :]) ** 2).sum(axis=2)
    assert d2.min(axis=1).sum() == pytest.approx(model.sse_trajectory_[-1])


def test_fit_predict_returns_memberships(blobs):
    X, _ = blobs
    U = CAFCM(n_clusters=4, random_state=0).fit_predict(X)
    assert U.shape == (160, 4)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)


def test_reproducible(blobs):
    X, _ = blobs
    a = CAFCM(n_clusters=4, random_state=9).fit(X).memberships_
    b = CAFCM(n_clusters=4, random_state=9).fit(X).memberships_
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m_start": 1.0},
        {"m_end": 1.0},
        {"m_start": 1.5, "m_end": 2.0},
        {"cooling_rate": 1.0},
        {"n_modules": 0},
    ],
)
def test_invalid_parameters_raise(blobs, kwargs):
    X, _ = blobs
    with pytest.raises(ValueError):
        CAFCM(n_clusters=4, **kwargs).fit(X)
