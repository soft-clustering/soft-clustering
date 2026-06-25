"""Unit and integration tests for KFCCL."""
import numpy as np
import pytest
from soft_clustering import KFCCL


@pytest.fixture
def X():
    rng = np.random.default_rng(6)
    return np.vstack([rng.normal([0, 0], 0.5, (15, 2)),
                      rng.normal([4, 4], 0.5, (15, 2))])


def test_fit_returns_labels(X):
    labels = KFCCL(n_clusters=2).fit(X)
    assert labels.shape == (30,)


def test_labels_in_range(X):
    labels = KFCCL(n_clusters=2).fit(X)
    assert set(labels).issubset({0, 1})


def test_membership_matrix_shape(X):
    model = KFCCL(n_clusters=2)
    model.fit(X)
    assert model.U.shape == (2, 30)


def test_kernel_matrix_computed(X):
    model = KFCCL(n_clusters=2)
    model.fit(X)
    assert model.K is not None
    assert model.K.shape == (30, 30)


def test_k3(X):
    labels = KFCCL(n_clusters=3).fit(X)
    assert set(labels).issubset({0, 1, 2})
