# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for CAFCM (collaborative annealing fuzzy c-means)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import CAFCM


def test_stalled_cooling_raises_in_bounded_child():
    code = textwrap.dedent("""
        import numpy as np
        from soft_clustering import CAFCM

        rate = float(np.nextafter(1.0, 0.0))
        assert 0 < rate < 1 and 1 + (2.0 - 1) * rate == 2.0
        try:
            CAFCM(n_clusters=2, cooling_rate=rate, random_state=0).fit(
                np.array([[0.0], [1.0], [10.0], [11.0]])
            )
        except ValueError as exc:
            assert "cooling schedule stalled" in str(exc)
        else:
            raise AssertionError("stalled schedule accepted")
        print("stalled schedule rejected")
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
    assert "stalled schedule rejected" in result.stdout


@pytest.mark.parametrize(
    "end,rate,expected",
    [
        (2.0, float(np.nextafter(1.0, 0.0)), [2.0]),
        (
            float(np.nextafter(2.0, 1.0)),
            float(np.nextafter(np.nextafter(1.0, 0.0), 0.0)),
            [2.0, float(np.nextafter(2.0, 1.0))],
        ),
    ],
)
def test_short_near_one_cooling_schedules(end, rate, expected):
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    model = CAFCM(
        n_clusters=2, m_start=2.0, m_end=end, cooling_rate=rate, random_state=0
    ).fit(X)
    np.testing.assert_array_equal(model.fuzzifier_schedule_, expected)
    assert np.isfinite(model.memberships_).all()


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
    np.testing.assert_allclose(
        schedule, np.maximum(1.0 + 0.7 ** np.arange(10), 1.05), rtol=0, atol=1e-15
    )
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
