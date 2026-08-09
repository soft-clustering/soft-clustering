"""Conformance tests for the SCPP estimator protocol.

Every estimator exported by ``soft_clustering`` is fitted here on a small,
valid input and checked against the contract documented in
``soft_clustering/_base.py``. These tests are what make the library's central
claim — one protocol across heterogeneous soft clustering methods — checkable
rather than asserted, and they fail automatically when a newly contributed
estimator deviates.

Adding an algorithm means adding one entry to ``CASES`` below; the shared
checks then apply to it without further work.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import soft_clustering as sc

ESTIMATORS = [
    n for n in sc.__all__ if n not in ("BaseSoftClusterer", "DEEP_ESTIMATORS")
]

K = 3
N = 24


# --------------------------------------------------------------------------
# Input builders, one per input modality the library accepts.
# --------------------------------------------------------------------------


def _features(n=N, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal(-3.0, 0.35, (n // 3, d)),
            rng.normal(0.0, 0.35, (n // 3, d)),
            rng.normal(3.0, 0.35, (n - 2 * (n // 3), d)),
        ]
    )


def _graph(n=N, seed=3):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) > 0.7).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def _docs():
    return [
        "fuzzy clustering membership degree",
        "cluster centroid distance metric",
        "graph community detection nodes",
        "network overlapping communities graph",
        "topic model document words",
        "document text word topic",
        "gaussian mixture probabilistic model",
        "probabilistic expectation maximization mixture",
    ]


def _ensemble(seed=7):
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(3):
        M = np.abs(rng.normal(size=(N, K)))
        mats.append(M / M.sum(axis=1, keepdims=True))
    return mats


# --------------------------------------------------------------------------
# Per-estimator construction. Each entry is (kwargs, fit-args callable).
# ``None`` marks an estimator excluded from the shared checks, with a reason.
# --------------------------------------------------------------------------

CASES: dict[str, tuple] = {
    # --- prototype / feature-matrix methods -------------------------------
    "FCM": ({"n_clusters": K, "random_state": 0}, lambda: (_features(),)),
    "PCM": ({"n_clusters": K, "random_state": 0}, lambda: (_features(),)),
    "GK": ({"n_clusters": K, "random_state": 0}, lambda: (_features(),)),
    "GMM": ({"n_clusters": K, "random_state": 0}, lambda: (_features(),)),
    "PFCM": ({"n_clusters": K, "random_state": 0}, lambda: (_features(),)),
    "CAFCM": ({"n_clusters": K}, lambda: (_features(),)),
    "CAFHFCM": ({"n_clusters": K}, lambda: (_features(),)),
    "ENTROPYFCM": ({"n_clusters": K}, lambda: (_features(),)),
    "AFCM": ({"n_clusters": K}, lambda: (_features(),)),
    "AFCMSimple": ({"n_clusters": K}, lambda: (_features(),)),
    "FCC": ({"n_clusters": K}, lambda: (_features(),)),
    "KFCM": ({"n_clusters": K}, lambda: (_features(),)),
    "KFCCL": ({"n_clusters": K}, lambda: (_features(),)),
    "ECM": ({"n_clusters": K}, lambda: (_features(),)),
    "RoughKMeans": ({"n_clusters": K}, lambda: (_features(),)),
    "MBMM": ({"n_clusters": K}, lambda: (np.abs(_features()) % 1.0,)),
    "SCM": ({}, lambda: (_features(),)),
    "SoftDBSCANGM": ({}, lambda: (_features(),)),
    "RPFKM": ({"n_clusters": K, "d": 2, "random_state": 0}, lambda: (_features(),)),
    # --- graph methods -----------------------------------------------------
    "BIGCLAM": ({"n_nodes": N, "n_clusters": K}, lambda: (_graph(),)),
    "BayesianNMF": ({"n_clusters": K}, lambda: (_graph(),)),
    "NOCD": (
        {"random_state": 1, "max_epochs": 3, "hidden_sizes": [8], "batch_size": 32},
        lambda: (sp.csr_matrix(_graph()), sp.eye(N, format="csr"), K),
    ),
    # --- document methods --------------------------------------------------
    "WBSC": ({}, lambda: (_docs(),)),
    "SISC": ({"n_clusters": 2}, lambda: (_docs(),)),
    "KMART": ({}, lambda: (_docs(),)),
    "PLSI": ({"n_clusters": 2, "max_iter": 5, "random_state": 0}, lambda: (_docs(),)),
    # --- ensemble methods --------------------------------------------------
    "SCSPA": ({"n_clusters": K}, lambda: (_ensemble(),)),
    "SHBGF": ({"n_clusters": K}, lambda: (_ensemble(),)),
    "SMCLA": ({"n_clusters": K}, lambda: (_ensemble(),)),
    # --- estimators outside the shared checks ------------------------------
    # Each entry states why; the count is asserted in test_exclusions_are_few.
    "AFCMAdaptive": None,  # segments a single image, not a sample matrix
    "SKFCM": None,  # requires the image shape alongside X
    "BGMM": None,  # consumes two aligned views (Xg, Xb)
    "SoftKSC": None,  # semi-supervised; needs labelled and unlabelled parts
    "SFCMEP": None,  # semi-supervised; needs a prior membership matrix
    "FeMIFuzzy": None,  # federated; consumes a list of client matrices
    "LDA": None,  # returns topic-word and doc-topic factors, not a partition
    "RDFKC": None,  # consumes image tensors; covered by tests/test_rdfkc.py
    "CDCGS": None,  # torch Module trained by the caller's own loop
    "DMoN": None,  # torch Module trained by the caller's own loop
    "MMSB": None,  # generative sampler rather than an estimator
}


COVERED = sorted(name for name, case in CASES.items() if case is not None)
EXCLUDED = sorted(name for name, case in CASES.items() if case is None)


def _fit(name):
    kwargs, make_args = CASES[name]
    model = getattr(sc, name)(**kwargs)
    return model.fit(*make_args())


# --------------------------------------------------------------------------
# The registry itself must stay in step with the public API.
# --------------------------------------------------------------------------


def test_every_estimator_is_registered():
    """No estimator may be added to the API without a conformance entry."""
    missing = sorted(set(ESTIMATORS) - set(CASES))
    assert not missing, f"estimators absent from CASES: {missing}"


def test_no_stale_registry_entries():
    stale = sorted(set(CASES) - set(ESTIMATORS))
    assert not stale, f"CASES names no longer exported: {stale}"


def test_exclusions_are_few():
    """Guard against the exclusion list quietly absorbing the whole library."""
    assert len(COVERED) >= 2 * len(EXCLUDED), (
        f"{len(EXCLUDED)} estimators excluded vs {len(COVERED)} covered; "
        "the protocol should apply to the clear majority of the library."
    )


# --------------------------------------------------------------------------
# The protocol.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", COVERED)
def test_fit_returns_self(name):
    kwargs, make_args = CASES[name]
    model = getattr(sc, name)(**kwargs)
    assert model.fit(*make_args()) is model


@pytest.mark.parametrize("name", COVERED)
def test_memberships_shape_and_dtype(name):
    model = _fit(name)
    U = model.memberships_
    assert U is not None, f"{name} did not populate memberships_"
    assert U.ndim == 2, f"{name}.memberships_ is not 2-D"
    assert np.issubdtype(U.dtype, np.floating)
    assert U.shape[1] == model.n_clusters


@pytest.mark.parametrize("name", COVERED)
def test_memberships_are_non_negative(name):
    U = _fit(name).memberships_
    assert np.all(U >= -1e-9), f"{name} produced negative memberships"


@pytest.mark.parametrize("name", COVERED)
def test_partition_constraint(name):
    model = _fit(name)
    if not model._partition_constrained:
        pytest.skip(f"{name} does not impose the partition constraint")
    U = model.memberships_
    if U.shape[1] == 0:
        pytest.skip(f"{name} discovered an empty partition on this input")
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", COVERED)
def test_labels_agree_with_argmax(name):
    model = _fit(name)
    U = model.memberships_
    if U.shape[1] == 0:
        pytest.skip(f"{name} discovered an empty partition on this input")
    np.testing.assert_array_equal(model.labels_, np.argmax(U, axis=1))
    assert model.labels_.shape == (U.shape[0],)


@pytest.mark.parametrize("name", COVERED)
def test_predict_and_predict_proba(name):
    model = _fit(name)
    np.testing.assert_array_equal(model.predict(), model.labels_)
    np.testing.assert_allclose(model.predict_proba(), model.memberships_)


@pytest.mark.parametrize("name", COVERED)
def test_centers_shape_when_present(name):
    model = _fit(name)
    if model.centers_ is None:
        pytest.skip(f"{name} is not prototype-based")
    assert model.centers_.ndim == 2


@pytest.mark.parametrize("name", COVERED)
def test_predict_before_fit_raises(name):
    kwargs, _ = CASES[name]
    model = getattr(sc, name)(**kwargs)
    with pytest.raises(RuntimeError):
        model.predict()


@pytest.mark.parametrize("name", [n for n in COVERED if "random_state" in CASES[n][0]])
def test_reproducible_under_fixed_seed(name):
    first = _fit(name).memberships_
    second = _fit(name).memberships_
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("name", COVERED)
def test_n_clusters_reports_the_fitted_partition(name):
    model = _fit(name)
    assert model.n_clusters == model.memberships_.shape[1]
