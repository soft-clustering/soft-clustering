"""Numerical edge cases across the estimator families.

The per-algorithm modules check that each method does the right thing on
well-behaved input. This module checks what happens at the boundaries, where
implementations usually break silently rather than loudly: coincident points
that send a distance to zero, more clusters than samples, a fuzzifier close to
one that sends an exponent to infinity, a covariance that is singular because
a cluster collapsed onto a line, and a graph with no edges.

The rule these tests enforce is uniform: **an estimator must either produce a
valid partition or raise.** Returning NaN, inf, or a membership matrix whose
rows do not mean what the protocol says they mean is the failure mode being
excluded. Where a method genuinely cannot handle an input, a raised exception
is the passing outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

import soft_clustering as sc

# Feature-matrix estimators, with a constructor that takes a cluster count.
FEATURE_ESTIMATORS = [
    "AFCM",
    "AFCMSimple",
    "CAFCM",
    "CAFHFCM",
    "ECM",
    "ENTROPYFCM",
    "EVCLUS",
    "FCC",
    "FCM",
    "GK",
    "GMM",
    "GathGeva",
    "KFCCL",
    "KFCM",
    "PCM",
    "PFCM",
    "RoughKMeans",
]


def build(name: str, k: int = 2):
    kwargs = {"n_clusters": k}
    if name in ("FCM", "GK", "GMM", "PCM", "PFCM", "GathGeva", "EVCLUS", "RoughKMeans"):
        kwargs["random_state"] = 0
    return getattr(sc, name)(**kwargs)


def assert_valid_partition(model, n_samples: int) -> None:
    U = model.memberships_
    assert U is not None, "no memberships produced"
    assert U.shape[0] == n_samples
    assert np.isfinite(U).all(), "membership matrix contains NaN or inf"
    assert np.all(U >= -1e-9), "negative memberships"
    if model._partition_constrained and U.shape[1] > 0:
        np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(model.labels_, np.argmax(U, axis=1))


# ---------------------------------------------------------------------------
# Degenerate geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FEATURE_ESTIMATORS)
def test_duplicate_points(name):
    """Coincident samples make at least one distance exactly zero.

    Every ratio-rule membership update divides by a distance, so this is the
    classic source of NaN rows in fuzzy clustering code.
    """
    rng = np.random.default_rng(0)
    base = np.vstack([rng.normal(-3, 0.3, (4, 3)), rng.normal(3, 0.3, (4, 3))])
    X = np.repeat(base, 5, axis=0)

    try:
        model = build(name).fit(X)
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip(f"{name} rejects duplicate points explicitly")
    assert_valid_partition(model, X.shape[0])


@pytest.mark.parametrize("name", FEATURE_ESTIMATORS)
def test_all_points_identical(name):
    """Every distance is zero, so nothing distinguishes the clusters."""
    X = np.ones((20, 3))
    try:
        model = build(name).fit(X)
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip(f"{name} rejects a constant dataset explicitly")
    assert_valid_partition(model, X.shape[0])


@pytest.mark.parametrize("name", FEATURE_ESTIMATORS)
def test_collinear_cluster_gives_singular_covariance(name):
    """A cluster confined to a line has a rank-deficient scatter matrix.

    Covariance-based methods (GK, Gath--Geva, GMM) must regularise rather than
    fail in LAPACK.
    """
    X = np.zeros((30, 3))
    X[:, 0] = np.linspace(-1, 1, 30)
    X[15:, 0] += 10.0

    try:
        model = build(name).fit(X)
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip(f"{name} rejects a degenerate covariance explicitly")
    assert_valid_partition(model, X.shape[0])


# ---------------------------------------------------------------------------
# Degenerate sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FEATURE_ESTIMATORS)
def test_more_clusters_than_samples(name):
    """K > n has no valid solution; raising is the only correct outcome."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 2))
    with pytest.raises((ValueError, IndexError, np.linalg.LinAlgError)):
        build(name, k=10).fit(X)


@pytest.mark.parametrize("name", FEATURE_ESTIMATORS)
def test_single_cluster(name):
    """K = 1 must give every sample full membership in the one cluster."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    try:
        model = build(name, k=1).fit(X)
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        pytest.skip(f"{name} requires at least two clusters")
    assert_valid_partition(model, X.shape[0])


# ---------------------------------------------------------------------------
# Extreme hyperparameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["FCM", "GK", "PCM", "PFCM", "GathGeva", "KFCM"])
def test_fuzzifier_close_to_one_does_not_overflow(name):
    """m -> 1 sends the ratio exponent 2/(m-1) to infinity.

    Evaluated naively, ``d ** -(2/(m-1))`` overflows to inf and the
    normalisation produces NaN. Every ratio rule in the library factors out
    the per-sample minimum distance before exponentiating, which caps every
    power at one.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.3, (20, 3)), rng.normal(3, 0.3, (20, 3))])
    model = getattr(sc, name)
    kwargs = {"n_clusters": 2, "m": 1.01}
    if name in ("FCM", "GK", "PCM", "PFCM", "GathGeva"):
        kwargs["random_state"] = 0
    try:
        fitted = model(**kwargs).fit(X)
    except (TypeError, ValueError):
        pytest.skip(f"{name} does not accept m=1.01")
    assert_valid_partition(fitted, X.shape[0])


@pytest.mark.parametrize("name", ["FCM", "GK", "PCM", "GathGeva"])
def test_large_fuzzifier(name):
    """m -> infinity flattens every membership towards 1/K."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.3, (20, 3)), rng.normal(3, 0.3, (20, 3))])
    fitted = getattr(sc, name)(n_clusters=2, m=20.0, random_state=0).fit(X)
    assert_valid_partition(fitted, X.shape[0])


# ---------------------------------------------------------------------------
# dtype and layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FEATURE_ESTIMATORS)
def test_float32_input(name):
    """Single-precision input must not change the output contract."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.3, (20, 3)), rng.normal(3, 0.3, (20, 3))]).astype(
        np.float32
    )
    try:
        model = build(name).fit(X)
    except (ValueError, TypeError) as exc:
        pytest.skip(f"{name} requires float64: {exc}")
    assert_valid_partition(model, X.shape[0])


@pytest.mark.parametrize("name", ["FCM", "GMM", "GK", "GathGeva"])
def test_fortran_ordered_input(name):
    """A non-C-contiguous view must give the same answer as a copy."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.3, (20, 3)), rng.normal(3, 0.3, (20, 3))])
    contiguous = build(name).fit(np.ascontiguousarray(X)).memberships_
    fortran = build(name).fit(np.asfortranarray(X)).memberships_
    np.testing.assert_allclose(contiguous, fortran, atol=1e-8)


def test_fortran_ordered_input_evclus():
    """EVCLUS agrees on the partition, and on the stress once it restarts.

    Its dissimilarities come from a Gram matrix, so whether a Fortran-ordered
    view gives a bit-identical result is a property of the BLAS rather than of
    this library: OpenBLAS reduces both layouts in the same order and the two
    fits agree to the last bit, while Accelerate does not and L-BFGS-B
    amplifies the last-bit difference into a different point on a very flat
    objective. Restarts absorb that: where a difference exists at all it is
    ~9e-4 with a single restart and ~1e-11 at the default n_init = 3.

    So the assertions here are the two that hold on either BLAS: the partition
    is identical, and the stress at the default n_init agrees to 1e-8 and is
    no more layout-sensitive than a single restart.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.3, (20, 3)), rng.normal(3, 0.3, (20, 3))])
    contiguous = build("EVCLUS").fit(np.ascontiguousarray(X))
    fortran = build("EVCLUS").fit(np.asfortranarray(X))
    np.testing.assert_array_equal(contiguous.labels_, fortran.labels_)
    restarted_gap = abs(contiguous.stress_ - fortran.stress_)
    assert restarted_gap < 1e-8

    # Restarting must not make the layout sensitivity worse. On a BLAS that
    # reduces both layouts identically both gaps are exactly zero.
    single_c = EVCLUS_single().fit(np.ascontiguousarray(X)).stress_
    single_f = EVCLUS_single().fit(np.asfortranarray(X)).stress_
    assert abs(single_c - single_f) >= restarted_gap


def EVCLUS_single():
    return sc.EVCLUS(n_clusters=2, n_init=1, random_state=0)


# ---------------------------------------------------------------------------
# Non-finite input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("name", ["FCM", "GMM", "GK", "GathGeva"])
def test_non_finite_input_does_not_produce_a_silent_partition(name, bad):
    """NaN or inf in the data must raise or propagate, never be ignored.

    The failure this excludes is an estimator that quietly returns a partition
    computed from corrupted distances.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.3, (20, 3)), rng.normal(3, 0.3, (20, 3))])
    X[0, 0] = bad

    try:
        model = build(name).fit(X)
    except (ValueError, np.linalg.LinAlgError):
        return  # raising is the preferred outcome
    U = model.memberships_
    assert not np.isfinite(U).all(), (
        f"{name} returned a fully finite partition from data containing {bad}; "
        "the corruption was silently absorbed"
    )


# ---------------------------------------------------------------------------
# Graph estimators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["BayesianNMF", "BIGCLAM", "MMSB"])
def test_empty_graph(name):
    """A graph with no edges carries no community structure."""
    A = np.zeros((12, 12))
    kwargs = {"n_clusters": 3}
    if name == "BIGCLAM":
        kwargs["n_nodes"] = 12
    if name == "MMSB":
        kwargs = {"n_blocks": 3, "random_state": 0, "max_iter": 5}
    try:
        model = getattr(sc, name)(**kwargs).fit(A)
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip(f"{name} rejects an edgeless graph explicitly")
    U = model.memberships_
    assert U is not None and np.isfinite(U).all()


@pytest.mark.parametrize("name", ["BayesianNMF", "MMSB"])
def test_complete_graph(name):
    """Every node connected to every other: one community, not a crash."""
    A = np.ones((12, 12))
    np.fill_diagonal(A, 0.0)
    kwargs = (
        {"n_blocks": 3, "random_state": 0, "max_iter": 5}
        if name == "MMSB"
        else {"n_clusters": 3}
    )
    model = getattr(sc, name)(**kwargs).fit(A)
    assert np.isfinite(model.memberships_).all()
