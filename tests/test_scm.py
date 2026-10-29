# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for SCM (Subtractive Clustering)."""

import numpy as np
import pytest

from soft_clustering import SCM


@pytest.fixture
def X():
    rng = np.random.default_rng(15)
    return np.vstack(
        [rng.normal([0, 0], 0.3, (20, 2)), rng.normal([5, 5], 0.3, (20, 2))]
    )


def test_fit_returns_self_and_stores_centers(X):
    model = SCM()
    assert model.fit(X) is model
    centers = model.centers_
    assert centers is not None
    assert centers.ndim == 2
    assert centers.shape[1] == 2


def test_centers_stored(X):
    model = SCM()
    model.fit(X)
    assert model.centers_ is not None


def test_at_least_one_center(X):
    model = SCM()
    centers = model.fit(X).centers_
    assert len(centers) >= 1


def test_centers_within_data_range(X):
    model = SCM()
    centers = model.fit(X).centers_
    for center in centers:
        assert np.all(center >= X.min(axis=0) - 1)
        assert np.all(center <= X.max(axis=0) + 1)


def test_different_ra_values(X):
    for ra in [0.3, 0.5, 0.8]:
        model = SCM(ra=ra)
        centers = model.fit(X).centers_
        assert centers.shape[1] == 2


def test_3d_input():
    rng = np.random.default_rng(16)
    X = rng.normal(size=(30, 3))
    model = SCM()
    centers = model.fit(X).centers_
    assert centers.shape[1] == 3


@pytest.mark.parametrize(
    "X, indices",
    [
        (np.array([[-2.0], [2.0]]), [0, 1]),
        (np.array([[-2.0, -3.0, -4.0], [-2.0, -3.0, -4.0], [2.0, 3.0, 4.0]]), [0, 2]),
    ],
    ids=["clusters-equal-samples", "features-equal-samples"],
)
def test_memberships_survive_center_dimension_collision_and_refit(X, indices):
    model = SCM()
    for offset in (0.0, 7.0):
        data = X + offset
        original = data.copy()
        data.setflags(write=False)
        if offset == 0.0:
            assert model.fit(X=data) is model
        else:
            memberships = model.fit_predict(X=data)
            np.testing.assert_array_equal(memberships, model.memberships_)

        np.testing.assert_array_equal(data, original)
        np.testing.assert_array_equal(model.center_indices_, indices)
        np.testing.assert_array_equal(model.centers_, original[indices])
        assert not np.shares_memory(model.centers_, data)
        assert model.n_clusters_ == len(indices)
        assert model.memberships_.shape == (len(data), len(indices))

        scaled = (original - original.min(axis=0)) / np.ptp(original, axis=0)
        squared_distances = ((scaled[:, None] - scaled[indices]) ** 2).sum(axis=2)
        degrees = np.exp(-4.0 / model.ra**2 * squared_distances)
        expected = degrees / degrees.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(model.memberships_, expected, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(model.predict_proba(), model.predict_proba(data))
        np.testing.assert_array_equal(model.labels_, expected.argmax(axis=1))
        np.testing.assert_array_equal(model.predict(), model.predict(data))
