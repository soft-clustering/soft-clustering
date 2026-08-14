# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

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


def _partial_labels(n=N, k=K):
    """Semi-supervised target: a few labelled samples per class, rest None."""
    labels = np.full(n, None, dtype=object)
    per_class = n // k
    for cluster in range(k):
        start = cluster * per_class
        labels[start : start + 2] = cluster
    return labels


def _ensemble(seed=7):
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(3):
        M = np.abs(rng.normal(size=(N, K)))
        mats.append(M / M.sum(axis=1, keepdims=True))
    return mats


def _image(h=6, w=8, seed=11):
    """A small grayscale image for the segmentation estimators."""
    rng = np.random.default_rng(seed)
    image = np.zeros((h, w))
    image[:, w // 2 :] = 1.0
    return image + rng.normal(0, 0.05, (h, w))


def _pixels(h=6, w=8, seed=11):
    """The same image as an (n_pixels, 1) feature matrix plus its shape."""
    return _image(h, w, seed).reshape(h * w, 1), (h, w)


def _two_views(n=N, seed=13):
    """Aligned Gaussian and Beta-distributed views, as BGMM consumes."""
    rng = np.random.default_rng(seed)
    gaussian = np.concatenate(
        [rng.normal(-2.0, 0.3, n // 2), rng.normal(2.0, 0.3, n - n // 2)]
    )
    beta = np.clip(
        np.concatenate([rng.beta(2.0, 6.0, n // 2), rng.beta(6.0, 2.0, n - n // 2)]),
        1e-3,
        1 - 1e-3,
    )
    return gaussian, beta


def _clients(seed=17):
    """Two federated clients over a shared feature space."""
    rng = np.random.default_rng(seed)
    names = ["f0", "f1", "f2"]
    first = np.vstack([rng.normal(-3, 0.3, (10, 3)), rng.normal(3, 0.3, (10, 3))])
    second = np.vstack([rng.normal(-3, 0.3, (8, 3)), rng.normal(3, 0.3, (8, 3))])
    return [first, second], [names, names]


def _semi_supervised(n=12, seed=19):
    """Labelled and unlabelled parts for the semi-supervised kernel method."""
    rng = np.random.default_rng(seed)
    labelled = np.vstack([rng.normal(-2, 0.3, (3, 3)), rng.normal(2, 0.3, (3, 3))])
    y = np.array([1, 1, 1, -1, -1, -1])
    unlabelled = rng.normal(0, 2.0, (n, 3))
    return labelled, y, unlabelled


def _autoencoder(n_features=8, latent=3):
    """Minimal encoder/decoder pair for the deep fuzzy k-means estimator."""
    import torch.nn as nn

    encoder = nn.Sequential(nn.Linear(n_features, latent))
    decoder = nn.Sequential(nn.Linear(latent, n_features))
    return encoder, decoder


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
    "GathGeva": ({"n_clusters": K, "random_state": 0}, lambda: (_features(),)),
    "EVCLUS": ({"n_clusters": K, "random_state": 0}, lambda: (_features(),)),
    "SFCMEP": ({"n_clusters": K}, lambda: (_features(), _partial_labels())),
    # --- graph methods -----------------------------------------------------
    "BIGCLAM": ({"n_nodes": N, "n_clusters": K}, lambda: (_graph(),)),
    "BayesianNMF": ({"n_clusters": K}, lambda: (_graph(),)),
    "MMSB": ({"n_blocks": K, "random_state": 0, "max_iter": 5}, lambda: (_graph(),)),
    "CDCGS": (
        {"n_clusters": K, "random_state": 0, "max_epochs": 20, "n_init": 1},
        lambda: (_graph(),),
    ),
    "DMoN": (
        {
            "in_channels": 4,
            "hidden_channels": 8,
            "n_clusters": K,
            "random_state": 0,
            "max_epochs": 10,
        },
        lambda: (_features(d=4), None, _graph()),
    ),
    "NOCD": (
        {"random_state": 1, "max_epochs": 3, "hidden_sizes": [8], "batch_size": 32},
        lambda: (sp.csr_matrix(_graph()), sp.eye(N, format="csr"), K),
    ),
    # --- document methods --------------------------------------------------
    "WBSC": ({}, lambda: (_docs(),)),
    "SISC": ({"n_clusters": 2}, lambda: (_docs(),)),
    "KMART": ({}, lambda: (_docs(),)),
    "PLSI": ({"n_clusters": 2, "max_iter": 5, "random_state": 0}, lambda: (_docs(),)),
    "LDA": ({"n_topics": 2, "max_iter": 5}, lambda: (_docs(),)),
    # --- image / segmentation methods --------------------------------------
    "AFCMAdaptive": ({"n_clusters": K, "max_iter": 5}, lambda: (_image(),)),
    "SKFCM": ({"n_clusters": K, "max_iter": 5}, lambda: _pixels()),
    # --- multi-view, federated, semi-supervised ----------------------------
    "BGMM": ({"n_clusters": 2}, lambda: _two_views()),
    "FeMIFuzzy": ({"random_state": 0}, lambda: _clients()),
    "SoftKSC": ({}, lambda: _semi_supervised()),
    # --- ensemble methods --------------------------------------------------
    "SCSPA": ({"n_clusters": K}, lambda: (_ensemble(),)),
    "SHBGF": ({"n_clusters": K}, lambda: (_ensemble(),)),
    "SMCLA": ({"n_clusters": K}, lambda: (_ensemble(),)),
}


def _register_rdfkc() -> None:
    """RDFKC needs an autoencoder, so its case is built lazily.

    Constructing torch modules at import time would make this module --- which
    every conformance test imports --- depend on the optional deep extra.
    """
    encoder, decoder = _autoencoder()
    CASES["RDFKC"] = (
        {
            "K": K,
            "encoder": encoder,
            "decoder": decoder,
            "random_state": 0,
            "max_iter": 2,
            "batch_size": 8,
        },
        lambda: (_features(d=8).astype("float32"),),
    )


_register_rdfkc()


COVERED = sorted(name for name, case in CASES.items() if case is not None)
EXCLUDED = sorted(name for name, case in CASES.items() if case is None)

#: Estimators whose formulation normalises memberships only for some inputs,
#: so observing normalised rows on the conformance input does not prove the
#: declaration wrong. Everything else declaring ``_partition_constrained =
#: False`` must actually produce unnormalised rows here --- see
#: ``test_partition_constraint_declaration_is_honest``, which exists because
#: three estimators once declared ``False`` while normalising unconditionally,
#: silently disabling the check below for themselves.
CONDITIONALLY_NORMALISED = frozenset(
    {
        "KMART",  # ART vigilance: a document may match zero or several categories
    }
)

#: Unseen data for the estimators that declare an out-of-sample rule, matching
#: the feature space each was fitted on in ``CASES``.
OUT_OF_SAMPLE = {
    "SoftKSC": lambda: np.random.default_rng(99).normal(0, 2.0, (7, 3)),
}


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


def test_no_estimator_is_excluded():
    """Every exported estimator is fitted by this suite.

    An earlier revision allowed exclusions "with a reason" and drifted to ten
    of forty, which meant the paper's claim that the conformance suite fits
    every exported estimator was false. Adding an estimator that the suite
    cannot construct now fails CI instead.
    """
    assert EXCLUDED == [], (
        f"{len(EXCLUDED)} estimator(s) excluded from the conformance suite: "
        f"{EXCLUDED}. Every exported estimator must have a runnable case."
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
def test_partition_constraint_declaration_is_honest(name):
    """An estimator may not understate its own guarantee.

    ``_partition_constrained = False`` switches off
    :func:`test_partition_constraint` for that estimator and makes the
    fuzziness indices report ``nan``. Declaring it while normalising
    unconditionally therefore hides a real invariant. Genuinely conditional
    cases are listed in ``CONDITIONALLY_NORMALISED``.
    """
    model = _fit(name)
    if model._partition_constrained or name in CONDITIONALLY_NORMALISED:
        pytest.skip(f"{name} declares the constraint or is conditionally normalised")
    U = model.memberships_
    if U is None or U.shape[1] == 0:
        pytest.skip(f"{name} produced no partition on this input")
    assert not np.allclose(U.sum(axis=1), 1.0, atol=1e-6), (
        f"{name} declares _partition_constrained = False but its memberships "
        "sum to one. Either set it to True, or add the estimator to "
        "CONDITIONALLY_NORMALISED with the input that breaks normalisation."
    )


@pytest.mark.parametrize("name", COVERED)
def test_predict_and_predict_proba(name):
    model = _fit(name)
    np.testing.assert_array_equal(model.predict(), model.labels_)
    np.testing.assert_allclose(model.predict_proba(), model.memberships_)


@pytest.mark.parametrize("name", COVERED)
def test_out_of_sample_predict_is_explicit(name):
    """``predict(X_new)`` must either work or raise --- never silently lie.

    Returning the stored training labels for unseen data hands back a
    partition of the wrong data, of the wrong length, with no indication that
    anything went wrong.
    """
    model = _fit(name)
    if model._supports_out_of_sample:
        unseen = OUT_OF_SAMPLE[name]()
        assert model.predict(unseen).shape == (unseen.shape[0],)
        assert model.predict_proba(unseen).shape[0] == unseen.shape[0]
    else:
        # The refusal happens before the argument is inspected, so any array
        # exercises it.
        with pytest.raises(NotImplementedError):
            model.predict(_features(n=7, seed=99))


@pytest.mark.parametrize("name", COVERED)
def test_get_params_round_trips_through_clone(name):
    """Parameter introspection must reconstruct an equivalent estimator."""
    sklearn_base = pytest.importorskip("sklearn.base")
    kwargs, _ = CASES[name]
    model = getattr(sc, name)(**kwargs)
    twin = sklearn_base.clone(model)

    assert type(twin) is type(model)
    original = model.get_params()
    copied = twin.get_params()
    assert set(copied) == set(original)
    # sklearn.clone deep-copies non-estimator parameters, so an object-valued
    # parameter (RDFKC's encoder and decoder) is a distinct but equivalent
    # instance. Compare the values that have meaningful equality.
    for key, value in original.items():
        if isinstance(value, (int, float, str, bool, tuple, type(None))):
            assert copied[key] == value, f"{name}.{key} did not round-trip"


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
