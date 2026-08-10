"""Equivalence tests for the optimized implementations.

The optimization study replaced some algorithm bodies with vectorised
implementations of the *same* mathematics. These tests pin that claim: each
optimized estimator is fitted alongside the preserved reference from
``optimization/original/scpp_original/`` on identical inputs, and the outputs
must agree to floating-point tolerance.

They are the regression guard for a specific failure mode — a future change
that speeds an estimator up by quietly altering what it computes.

The reference package lives outside the distribution, so these tests skip when
it is unavailable (for instance when running against an installed wheel rather
than a checkout).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "optimization" / "original"
if _REFERENCE_DIR.is_dir() and str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))

scpp_original = pytest.importorskip(
    "scpp_original",
    reason="preserved reference implementations are not on the path",
)

import soft_clustering as sc  # noqa: E402

# Absolute tolerance for membership matrices. The measured worst case across
# the study was 2.2e-14 (see optimization/benchmarks/results.json); 1e-9 leaves
# room for platform BLAS differences without admitting a real change.
ATOL = 1e-9


def _blobs(n=120, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    per = n // k
    parts = [rng.normal(3.0 * i, 0.35, (per, d)) for i in range(k - 1)]
    parts.append(rng.normal(3.0 * (k - 1), 0.35, (n - per * (k - 1), d)))
    return np.vstack(parts)


def _unit_blobs(n=200, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((n, d)), 1e-3, 1 - 1e-3)


def _fit_pair(name, X, *, kwargs=None, global_seed=0):
    """Fit the optimized and reference builds under identical conditions."""
    kwargs = kwargs or {}
    fitted = []
    for module in (sc, scpp_original):
        estimator = getattr(module, name)(**kwargs)
        # Some estimators draw initial parameters from NumPy's global RNG
        # rather than a random_state argument; seeding it here is what makes
        # the comparison deterministic.
        np.random.seed(global_seed)
        estimator.fit(X)
        fitted.append(estimator)
    return fitted


class TestSoftDBSCANGM:
    """Vectorised Mahalanobis distances and closed-form membership update."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_memberships_match_reference(self, seed):
        X = _blobs(seed=seed)
        opt, ref = _fit_pair("SoftDBSCANGM", X, global_seed=seed)
        assert opt.memberships_.shape == ref.memberships_.shape
        np.testing.assert_allclose(opt.memberships_, ref.memberships_, atol=ATOL)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_labels_and_centres_match_reference(self, seed):
        X = _blobs(seed=seed)
        opt, ref = _fit_pair("SoftDBSCANGM", X, global_seed=seed)
        assert np.array_equal(opt.labels_, ref.labels_)
        assert opt.n_clusters == ref.n_clusters
        # This estimator publishes its prototypes as ``centers`` (no trailing
        # underscore), which is not one of the names
        # ``BaseSoftClusterer._centers_attrs`` searches, so ``centers_`` is None
        # on both builds. That protocol gap predates the optimization and is
        # recorded in the study report; the comparison uses the attribute the
        # estimator actually populates.
        assert opt.centers_ is None and ref.centers_ is None
        np.testing.assert_allclose(opt.centers, ref.centers, atol=1e-8)

    def test_invariants(self):
        X = _blobs()
        opt, _ = _fit_pair("SoftDBSCANGM", X)
        U = opt.memberships_
        assert np.isfinite(U).all()
        assert (U >= 0).all()
        assert np.array_equal(np.argmax(U, axis=1), opt.labels_)

    def test_membership_update_is_normalised(self):
        """The closed form must reproduce the reference's normalisation: the
        ratio-sum definition makes each row sum to one by construction."""
        X = _blobs()
        opt, _ = _fit_pair("SoftDBSCANGM", X)
        np.testing.assert_allclose(opt.memberships_.sum(axis=1), 1.0, atol=1e-12)

    def test_small_fuzzifier_does_not_overflow(self):
        """The per-sample rescaling exists so that a small ``m`` — a large
        exponent — cannot overflow before normalisation."""
        X = _blobs(n=60)
        model = sc.SoftDBSCANGM(m=1.05, max_iter=5).fit(X)
        assert np.isfinite(model.memberships_).all()
        np.testing.assert_allclose(model.memberships_.sum(axis=1), 1.0, atol=1e-12)


class TestMBMM:
    """Direct scipy.special Beta log-density in place of the frozen
    distribution, and a vectorised M-step."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_responsibilities_match_reference(self, seed):
        X = _unit_blobs(seed=seed)
        opt, ref = _fit_pair("MBMM", X, global_seed=seed)
        np.testing.assert_allclose(opt.resp, ref.resp, atol=ATOL)
        np.testing.assert_allclose(opt.memberships_, ref.memberships_, atol=ATOL)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_parameters_match_reference(self, seed):
        X = _unit_blobs(seed=seed)
        opt, ref = _fit_pair("MBMM", X, global_seed=seed)
        np.testing.assert_allclose(opt.alpha, ref.alpha, atol=1e-8)
        np.testing.assert_allclose(opt.beta, ref.beta, atol=1e-8)
        np.testing.assert_allclose(opt.weights, ref.weights, atol=1e-10)

    def test_labels_match_reference(self):
        X = _unit_blobs()
        opt, ref = _fit_pair("MBMM", X)
        assert np.array_equal(opt.labels_, ref.labels_)

    def test_log_density_matches_scipy(self):
        """The vectorised Beta log-density must equal the frozen distribution
        it replaced, on the open interval the model assumes."""
        from scipy.stats import beta as beta_dist

        from soft_clustering._mbmm import MBMM

        rng = np.random.default_rng(0)
        X = np.clip(rng.random((50, 4)), 1e-3, 1 - 1e-3)

        model = MBMM(n_components=2)
        model.weights = np.array([0.4, 0.6])
        model.alpha = rng.uniform(1.0, 3.0, size=(2, 4))
        model.beta = rng.uniform(1.0, 3.0, size=(2, 4))

        got = model._log_prob(X)

        expected = np.empty_like(got)
        for k in range(2):
            acc = np.log(model.weights[k] + 1e-10)
            for d in range(4):
                acc = acc + beta_dist.logpdf(
                    X[:, d], model.alpha[k, d], model.beta[k, d]
                )
            expected[:, k] = acc

        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_reference_package_is_not_importable_from_the_distribution():
    """The preserved originals must stay out of the production import path."""
    assert not (
        Path(sc.__file__).parent / "original"
    ).exists(), "reference implementations must not ship inside soft_clustering/"
    assert "scpp_original" not in sc.__file__
