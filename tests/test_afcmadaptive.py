# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for AFCMAdaptive (image-based FCM)."""

import numpy as np
import pytest

from soft_clustering import AFCMAdaptive


@pytest.fixture
def image():
    rng = np.random.default_rng(13)
    return rng.random((20, 20)).astype(float)


def test_fit_runs(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    assert model.centers is not None


def test_predict_shape(image):
    """predict() returns the flat labelling the estimator protocol requires."""
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    labels = model.predict()
    assert labels.shape == (400,)


def test_label_map_returns_image_layout(image):
    """label_map() is the same assignment reshaped back onto the pixel grid."""
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    label_map = model.label_map()
    assert label_map.shape == (20, 20)
    np.testing.assert_array_equal(label_map.ravel(), model.predict())


def test_memberships_are_pixel_major(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    assert model.memberships_.shape == (400, 3)
    np.testing.assert_allclose(model.memberships_.sum(axis=1), 1.0, atol=1e-6)
    # The flat view and the image-shaped array hold the same numbers.
    np.testing.assert_allclose(
        model.memberships_.reshape(20, 20, 3), model.get_membership()
    )


def test_labels_in_range(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    labels = model.predict()
    assert set(labels.flat).issubset({0, 1, 2})


def test_membership_shape(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    M = model.get_membership()
    assert M.shape == (20, 20, 3)


def test_membership_sums_to_one(image):
    model = AFCMAdaptive(n_clusters=3, max_iter=5)
    model.fit(image)
    M = model.get_membership()
    np.testing.assert_allclose(M.sum(axis=2), 1.0, atol=1e-5)


def test_k2(image):
    model = AFCMAdaptive(n_clusters=2, max_iter=5)
    model.fit(image)
    assert model.predict().shape == (400,)
    assert model.label_map().shape == (20, 20)
