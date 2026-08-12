"""Tests for Gath--Geva fuzzy maximum-likelihood estimation clustering."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from soft_clustering import FCM, GathGeva


def blobs(n_per=40, d=4, k=3, spread=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(3.0 * i, spread, (n_per, d)) for i in range(k)])
    return X, np.repeat(np.arange(k), n_per)


def anisotropic(n_per=80, seed=0):
    """Two elongated clusters with orthogonal principal axes.

    Gath--Geva models a full covariance per cluster; fuzzy c-means assumes
    spherical clusters. This is the configuration that distinguishes them.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n_per, 2)) @ np.array([[3.0, 0.0], [0.0, 0.25]])
    b = rng.normal(0, 1, (n_per, 2)) @ np.array([[0.25, 0.0], [0.0, 3.0]]) + 4.0
    return np.vstack([a, b]), np.repeat([0, 1], n_per)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    X, _ = blobs()
    model = GathGeva(n_clusters=3, random_state=0)
    assert model.fit(X) is model


def test_output_shapes():
    X, _ = blobs()
    model = GathGeva(n_clusters=3, random_state=0).fit(X)
    assert model.memberships_.shape == (X.shape[0], 3)
    assert model.centers_.shape == (3, X.shape[1])
    assert model.covariances_.shape == (3, X.shape[1], X.shape[1])
    assert model.priors_.shape == (3,)


def test_memberships_sum_to_one():
    X, _ = blobs()
    U = GathGeva(n_clusters=3, random_state=0).fit(X).memberships_
    assert np.all(U >= 0)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)


def test_labels_are_argmax():
    X, _ = blobs()
    model = GathGeva(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(model.labels_, np.argmax(model.memberships_, axis=1))


def test_priors_sum_to_one():
    X, _ = blobs()
    priors = GathGeva(n_clusters=3, random_state=0).fit(X).priors_
    np.testing.assert_allclose(priors.sum(), 1.0, atol=1e-9)


def test_covariances_are_symmetric_positive_definite():
    X, _ = blobs()
    cov = GathGeva(n_clusters=3, random_state=0).fit(X).covariances_
    for matrix in cov:
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(matrix) > 0)


def test_reproducible_under_fixed_seed():
    X, _ = blobs()
    first = GathGeva(n_clusters=3, random_state=1).fit(X).memberships_
    second = GathGeva(n_clusters=3, random_state=1).fit(X).memberships_
    np.testing.assert_allclose(first, second)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GathGeva(n_clusters=3).predict()


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------


def test_recovers_well_separated_blobs():
    X, y = blobs(seed=2)
    model = GathGeva(n_clusters=3, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.95


def test_beats_fcm_on_anisotropic_clusters():
    """The point of the exponential distance: per-cluster covariance."""
    X, y = anisotropic(seed=3)
    gg = adjusted_rand_score(y, GathGeva(n_clusters=2, random_state=0).fit(X).labels_)
    fcm = adjusted_rand_score(y, FCM(n_clusters=2, random_state=0).fit(X).labels_)
    assert gg > fcm


def test_fcm_init_beats_random_init():
    """Documents why `init='fcm'` is the default, as Gath and Geva prescribe."""
    X, y = blobs(seed=2)
    with_fcm = adjusted_rand_score(
        y, GathGeva(n_clusters=3, init="fcm", random_state=0).fit(X).labels_
    )
    with_random = adjusted_rand_score(
        y, GathGeva(n_clusters=3, init="random", random_state=0).fit(X).labels_
    )
    assert with_fcm > with_random


def test_converges_before_the_iteration_cap():
    X, _ = blobs(seed=2)
    model = GathGeva(n_clusters=3, random_state=0, max_iter=200).fit(X)
    assert model.n_iter_ < 200


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_duplicate_points_stay_finite(self):
        """Coincident samples drive the exponential distance to zero."""
        X, _ = blobs(n_per=4)
        X = np.repeat(X, 5, axis=0)
        U = GathGeva(n_clusters=2, random_state=0).fit(X).memberships_
        assert np.isfinite(U).all()
        np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)

    def test_singular_cluster_is_regularised(self):
        """A cluster confined to a line has a rank-deficient covariance."""
        X = np.zeros((30, 3))
        X[:, 0] = np.linspace(-1, 1, 30)
        X[15:, 0] += 10.0
        U = GathGeva(n_clusters=2, random_state=0).fit(X).memberships_
        assert np.isfinite(U).all()

    def test_small_fuzzifier_does_not_overflow(self):
        X, _ = blobs(seed=4)
        U = GathGeva(n_clusters=3, m=1.05, random_state=0).fit(X).memberships_
        assert np.isfinite(U).all()
        np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-9)

    def test_single_cluster(self):
        X, _ = blobs()
        U = GathGeva(n_clusters=1, random_state=0).fit(X).memberships_
        np.testing.assert_allclose(U, 1.0)

    def test_float32_input_is_accepted(self):
        X, _ = blobs()
        U = GathGeva(n_clusters=3, random_state=0).fit(X.astype(np.float32))
        assert U.memberships_.dtype == np.float64


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"n_clusters": 0}, {"m": 1.0}, {"m": 0.5}, {"max_iter": 0}, {"reg_covar": -1.0}],
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        GathGeva(**kwargs)


def test_invalid_init_raises():
    with pytest.raises(ValueError, match="init must be"):
        GathGeva(init="nope")


def test_one_dimensional_input_raises():
    with pytest.raises(ValueError, match="2-D"):
        GathGeva(n_clusters=2).fit(np.arange(10.0))


def test_more_clusters_than_samples_raises():
    X, _ = blobs(n_per=1, k=2)
    with pytest.raises(ValueError, match="smaller than"):
        GathGeva(n_clusters=10).fit(X)
