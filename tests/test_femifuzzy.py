"""Unit and integration tests for FeMIFuzzy (Federated Fuzzy Clustering with MI)."""
import numpy as np
import pytest
from soft_clustering import FeMIFuzzy


@pytest.fixture
def clients_features():
    rng = np.random.default_rng(39)
    features_all = ["f1", "f2", "f3", "f4"]
    clients = [
        rng.normal(size=(8, 4)),
        rng.normal(size=(8, 4)),
    ]
    features = [features_all, features_all]
    return clients, features


def test_fit_predict_returns_list(clients_features):
    clients, features = clients_features
    model = FeMIFuzzy(random_state=0, max_iter=2)
    result = model.fit_predict(clients, features)
    assert isinstance(result, list)


def test_global_centroids_shape(clients_features):
    clients, features = clients_features
    model = FeMIFuzzy(random_state=0, max_iter=2)
    result = model.fit_predict(clients, features)
    # Each centroid should be a 2-D array (Sammon maps to 2D)
    assert len(result) >= 1
    for centroid in result:
        assert centroid.shape[-1] == 2


def test_no_common_features_raises():
    rng = np.random.default_rng(40)
    clients = [rng.normal(size=(5, 2)), rng.normal(size=(5, 2))]
    features = [["a", "b"], ["c", "d"]]
    model = FeMIFuzzy(random_state=0, max_iter=2)
    with pytest.raises(ValueError):
        model.fit_predict(clients, features)


def test_partial_overlap():
    rng = np.random.default_rng(41)
    clients = [rng.normal(size=(5, 3)), rng.normal(size=(5, 3))]
    features = [["a", "b", "c"], ["b", "c", "d"]]
    model = FeMIFuzzy(random_state=0, max_iter=2)
    result = model.fit_predict(clients, features)
    assert isinstance(result, list)
