# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests specific to SHBGF; behaviour shared by the consensus estimators is in
``tests/test_consensus.py``."""

import numpy as np
import pytest

from soft_clustering import SHBGF


@pytest.fixture
def soft_memberships():
    rng = np.random.default_rng(29)
    mats = []
    for _ in range(3):
        m = np.abs(rng.normal(size=(30, 3)))
        m /= m.sum(axis=1, keepdims=True)
        mats.append(m)
    return mats


def test_fit_predict_returns_memberships(soft_memberships):
    model = SHBGF(n_clusters=3, random_state=0)
    U = model.fit_predict(soft_memberships)
    assert U.shape == (30, 3)
    np.testing.assert_array_equal(U, model.memberships_)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-12)


def test_k2(soft_memberships):
    assert SHBGF(n_clusters=2, random_state=0).fit_predict(soft_memberships).shape == (
        30,
        2,
    )


def test_more_clusters_than_available_raises(soft_memberships):
    with pytest.raises(ValueError, match="exceeds"):
        SHBGF(n_clusters=100).fit_predict(soft_memberships)


def test_partitions_the_row_normalised_concatenation(soft_memberships):
    mats = [m.copy() for m in soft_memberships]
    for m in mats:
        m[1] = m[0] * 3.0  # identical after row normalisation
    model = SHBGF(n_clusters=3, random_state=0)
    model.fit_predict(mats)
    assert model.labels_[0] == model.labels_[1]
