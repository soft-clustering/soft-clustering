"""Unit and integration tests for SHBGF."""
import numpy as np
import pytest
from soft_clustering import SHBGF


@pytest.fixture
def soft_memberships():
    rng = np.random.default_rng(27)
    mats = []
    for _ in range(3):
        m = np.abs(rng.normal(size=(30, 3)))
        m /= m.sum(axis=1, keepdims=True)
        mats.append(m)
    return mats


def test_fit_predict_shape(soft_memberships):
    labels = SHBGF(n_clusters=3).fit_predict(soft_memberships)
    assert labels.shape == (30,)


def test_labels_in_range(soft_memberships):
    labels = SHBGF(n_clusters=3).fit_predict(soft_memberships)
    assert set(labels).issubset({0, 1, 2})


def test_k2(soft_memberships):
    labels = SHBGF(n_clusters=2).fit_predict(soft_memberships)
    assert labels.shape == (30,)


def test_single_partition():
    rng = np.random.default_rng(28)
    m = np.abs(rng.normal(size=(20, 2)))
    m /= m.sum(axis=1, keepdims=True)
    labels = SHBGF(n_clusters=2).fit_predict([m])
    assert labels.shape == (20,)
