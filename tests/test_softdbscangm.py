"""Unit and integration tests for SoftDBSCANGM."""

import numpy as np
import pytest
from soft_clustering import SoftDBSCANGM


@pytest.fixture
def X():
    rng = np.random.default_rng(19)
    return np.vstack(
        [rng.normal([0, 0], 0.3, (20, 2)), rng.normal([4, 4], 0.3, (20, 2))]
    )


def test_fit_runs(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    assert model.U is not None


def test_membership_shape(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    M = model.get_membership()
    assert M.ndim == 2
    assert M.shape[0] == 40


def test_membership_rows_sum_to_one(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    M = model.get_membership()
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-5)


def test_predict_shape(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    labels = model.predict()
    assert labels.shape == (40,)


def test_centers_shape(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    k = model.U.shape[1]
    assert model.centers.shape == (k, 2)


def test_labels_stored(X):
    model = SoftDBSCANGM(eps=1.0, min_samples=3, max_iter=5)
    model.fit(X)
    assert model.labels_ is not None
    assert model.labels_.shape == (40,)
