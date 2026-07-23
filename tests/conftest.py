"""Shared fixtures for SCPP unit and integration tests."""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def blobs_2d():
    """Two well-separated 2-D clusters, 60 samples total."""
    rng = np.random.default_rng(0)
    X1 = rng.normal([0, 0], 0.3, (30, 2))
    X2 = rng.normal([4, 4], 0.3, (30, 2))
    return np.vstack([X1, X2])


@pytest.fixture
def blobs_3d():
    """Three 3-D clusters."""
    rng = np.random.default_rng(1)
    X1 = rng.normal([0, 0, 0], 0.3, (20, 3))
    X2 = rng.normal([5, 0, 0], 0.3, (20, 3))
    X3 = rng.normal([0, 5, 0], 0.3, (20, 3))
    return np.vstack([X1, X2, X3])


@pytest.fixture
def soft_memberships_list():
    """Three (30×3) soft membership matrices for consensus tests."""
    rng = np.random.default_rng(7)
    mats = []
    for _ in range(3):
        m = np.abs(rng.normal(size=(30, 3)))
        m /= m.sum(axis=1, keepdims=True)
        mats.append(m)
    return mats


@pytest.fixture
def small_adjacency():
    """Symmetric binary 10×10 adjacency matrix."""
    rng = np.random.default_rng(3)
    A = (rng.random((10, 10)) > 0.6).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


@pytest.fixture
def text_docs():
    return [
        "machine learning clustering algorithm data",
        "fuzzy membership cluster centroid data",
        "neural network deep learning model",
        "graph community detection network nodes",
        "document text word topic model",
        "probabilistic mixture gaussian expectation",
        "soft assignment membership degree cluster",
        "kernel function similarity distance metric",
    ]
