# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Equivalence tests for the optimized implementations.

The optimization study replaced some algorithm bodies with vectorised
implementations of the *same* mathematics. These tests pin that claim: each
optimized estimator is fitted alongside the preserved reference from
``optimization/original/scpp_original/`` on identical inputs. Training updates
and unchanged outputs must agree to floating-point tolerance. SoftDBSCANGM now
also exposes its existing prototypes as ``centers_``; MBMM's corrected terminal
posterior is checked independently under the unchanged final parameters rather
than against the reference's stale responsibilities. KMART's inclusive
vigilance is checked against a scalar rule oracle, with archived comparisons
retained where both threshold formulas make the same decisions.

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
from scipy.special import softmax
from scipy.stats import beta as beta_dist

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


_WORDS = (
    "fuzzy clustering membership degree centroid distance metric kernel "
    "graph community detection nodes network overlapping topic document "
    "word probabilistic mixture gaussian expectation maximization soft "
    "assignment spectral possibilistic evidential rough consensus ensemble"
).split()


def _documents(n=200, length=8, seed=5):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(_WORDS, size=length, replace=True)) for _ in range(n)]


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
        assert opt.n_clusters_ == ref.n_clusters
        # The reference lacks canonical centers; the repaired estimator exposes
        # the same prototypes without changing their legacy values.
        assert ref.centers_ is None
        np.testing.assert_allclose(opt.centers, ref.centers, atol=1e-8)
        np.testing.assert_array_equal(opt.centers_, opt.centers)

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
    distribution, a vectorised M-step, and a corrected terminal posterior."""

    @staticmethod
    def _final_posterior(X, ref):
        log_joint = np.column_stack(
            [
                np.log(ref.weights[k] + 1e-10)
                + beta_dist.logpdf(X, ref.alpha[k], ref.beta[k]).sum(axis=1)
                for k in range(ref.n_components)
            ]
        )
        return softmax(log_joint, axis=1)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_responsibilities_match_final_parameter_posterior(self, seed):
        X = _unit_blobs(seed=seed)
        opt, ref = _fit_pair("MBMM", X, global_seed=seed)
        expected = self._final_posterior(X, ref)
        np.testing.assert_allclose(opt.resp, expected, atol=ATOL)
        np.testing.assert_allclose(opt.memberships_, expected, atol=ATOL)

    @pytest.mark.parametrize("seed", [0, 1])
    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"max_iter": 1}, {"max_iter": 30, "tol": 1e6}, {"max_iter": 0}],
        ids=["default", "one-update", "early-stop", "initial-parameters"],
    )
    def test_parameters_match_reference(self, seed, kwargs):
        X = _unit_blobs(seed=seed)
        opt, ref = _fit_pair("MBMM", X, global_seed=seed, kwargs=kwargs)
        np.testing.assert_allclose(opt.alpha, ref.alpha, atol=1e-8)
        np.testing.assert_allclose(opt.beta, ref.beta, atol=1e-8)
        np.testing.assert_allclose(opt.weights, ref.weights, atol=1e-10)

    def test_labels_match_final_parameter_posterior(self):
        X = _unit_blobs()
        opt, ref = _fit_pair("MBMM", X)
        np.testing.assert_array_equal(
            opt.labels_, self._final_posterior(X, ref).argmax(axis=1)
        )

    def test_log_density_matches_scipy(self):
        """The vectorised Beta log-density must equal the frozen distribution
        it replaced, on the open interval the model assumes."""
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


# KFCCL was part of the optimisation study, but the implementation it
# optimised was found in the 0.2.0 correctness audit not to cluster at all
# (every membership 1/K on well-separated data) and has been replaced. There is
# no longer an optimised build of the same algorithm to compare against the
# preserved reference, so its equivalence tests were removed. See
# optimization/README.md, "Status after the 0.2.0 correctness audit".


def _kmart_scalar(vectors, rho, rate):
    """Independent sequential rule on known count vectors, without vectorization."""
    prototypes, clusters, decisions = [], [], []
    for i, vector in enumerate(vectors):
        norm = sum(vector)
        matched = False
        for j, prototype in enumerate(prototypes):
            intersection = [min(x, p) for x, p in zip(vector, prototype)]
            overlap = sum(intersection)
            score = overlap / norm if norm > 0 else 0.0
            decisions.append((score, overlap / (norm + 1e-9)))
            if score >= rho:
                matched = True
                clusters[j].add(i)
                prototypes[j] = [
                    rate * x + (1 - rate) * p for x, p in zip(intersection, prototype)
                ]
        if not matched:
            prototypes.append(list(vector))
            clusters.append({i})
    memberships = np.zeros((len(vectors), len(clusters)))
    for j, members in enumerate(clusters):
        for i in members:
            memberships[i, j] = 1
    return memberships, prototypes, clusters, np.array(decisions)


def _known_document_counts(docs):
    """Count the non-stopword vocabulary used by the randomized fixtures."""
    tokens = [doc.split() for doc in docs]
    vocabulary = sorted({word for row in tokens for word in row})
    return [[float(row.count(word)) for word in vocabulary] for row in tokens]


class TestKMART:
    """Vectorized vigilance with the corrected inclusive threshold contract."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_memberships_match_scalar_rule(self, seed):
        docs = _documents(seed=seed)
        opt = sc.KMART()
        opt_U = opt.fit_predict(docs)
        expected, prototypes, clusters, _ = _kmart_scalar(
            _known_document_counts(docs), 0.5, 1.0
        )
        np.testing.assert_array_equal(opt_U, expected)
        np.testing.assert_array_equal(opt.prototypes_, prototypes)
        assert opt.clusters_ == clusters

    def test_prototypes_and_clusters_match_scalar_rule(self):
        docs = _documents()
        opt = sc.KMART()
        opt.fit_predict(docs)
        _, prototypes, clusters, _ = _kmart_scalar(
            _known_document_counts(docs), 0.5, 1.0
        )
        assert opt.clusters_ == clusters
        assert len(opt.prototypes_) == len(prototypes)
        for got, expected in zip(opt.prototypes_, prototypes):
            np.testing.assert_array_equal(got, expected)

    def test_prototypes_are_published_as_a_list_of_vectors(self):
        """The buffer is an implementation detail; the documented attribute is
        a list of per-cluster arrays."""
        docs = _documents(n=60)
        model = sc.KMART()
        model.fit_predict(docs)
        assert isinstance(model.prototypes_, list)
        assert all(isinstance(p, np.ndarray) and p.ndim == 1 for p in model.prototypes_)
        assert len(model.prototypes_) == len(model.clusters_)

    def test_partial_learning_rate_matches_scalar_rule(self):
        """learning_rate < 1 exercises the convex-combination update path."""
        docs = _documents(n=120)
        opt = sc.KMART(vigilance_param=0.3, learning_rate=0.4)
        expected, prototypes, clusters, _ = _kmart_scalar(
            _known_document_counts(docs), 0.3, 0.4
        )
        np.testing.assert_array_equal(opt.fit_predict(docs), expected)
        np.testing.assert_array_equal(opt.prototypes_, prototypes)
        assert opt.clusters_ == clusters

    @pytest.mark.parametrize("rate", [1.0, 0.4])
    def test_nonboundary_training_matches_archived_reference(self, rate):
        docs = [
            "apple apple",
            "banana banana",
            "apple apple",
            "banana banana",
            "apple banana",
            "apple banana",
        ]
        vectors = [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
        expected, prototypes, clusters, decisions = _kmart_scalar(vectors, 0.4, rate)
        # Every decision has a strict margin for both the rule and archived epsilon.
        assert np.min(np.abs(decisions - 0.4)) > 0.09
        np.testing.assert_array_equal(decisions[:, 0] >= 0.4, decisions[:, 1] >= 0.4)
        np.testing.assert_array_equal(
            expected, [[1, 0], [0, 1], [1, 0], [0, 1], [1, 1], [1, 1]]
        )
        opt = sc.KMART(vigilance_param=0.4, learning_rate=rate)
        ref = scpp_original.KMART(vigilance_param=0.4, learning_rate=rate)
        np.testing.assert_array_equal(
            opt.fit_predict(docs), ref.fit_predict(docs).toarray()
        )
        np.testing.assert_array_equal(opt.memberships_, expected)
        assert opt.clusters_ == ref.clusters_ == clusters
        np.testing.assert_array_equal(opt.prototypes_, ref.prototypes_)
        np.testing.assert_array_equal(opt.prototypes_, prototypes)
        assert opt.cluster_words_ == ref.cluster_words_


class TestKFCM:
    """Matrix-form Gaussian kernel in place of the scalar helper."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_memberships_match_reference(self, seed):
        X = _blobs(seed=seed)
        opt, ref = _fit_pair("KFCM", X, kwargs={"n_clusters": 3}, global_seed=seed)
        np.testing.assert_allclose(opt.memberships_, ref.memberships_, atol=ATOL)
        np.testing.assert_allclose(opt.V, ref.V, atol=1e-8)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_labels_match_reference(self, seed):
        X = _blobs(seed=seed)
        opt, ref = _fit_pair("KFCM", X, kwargs={"n_clusters": 3}, global_seed=seed)
        assert np.array_equal(opt.labels_, ref.labels_)

    def test_kernel_matrix_matches_the_scalar_helper(self):
        """The batched kernel must agree with the scalar ``_gaussian_kernel``
        it replaced, including the sqrt round trip.

        Agreement is close but not bit-for-bit: ``np.linalg.norm`` reduces via
        a BLAS dot product, which need not sum in the same order as ``np.sum``
        over the last axis. The resulting 1-ULP difference in the squared
        distance is amplified by ``exp`` in proportion to the distance, so the
        bound is stated as a relative tolerance on the kernel value.
        """
        X = _blobs(n=30, d=4)
        model = sc.KFCM(n_clusters=3, sigma=1.3)
        centers = X[:3]
        got = model._kernel_matrix(X, centers)
        expected = np.array(
            [[model._gaussian_kernel(x, c) for c in centers] for x in X]
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)

    def test_initialisation_consumes_the_same_random_draws(self):
        """K-Means++ must take one randint and one rand per extra center, in
        that order — otherwise every downstream value diverges."""
        X = _blobs(n=50, d=4)
        for seed in (0, 3):
            np.random.seed(seed)
            got = sc.KFCM(n_clusters=4)._initialize_centers_kmeans_pp(X)
            np.random.seed(seed)
            expected = scpp_original.KFCM(n_clusters=4)._initialize_centers_kmeans_pp(X)
            np.testing.assert_array_equal(got, expected)
            # The global RNG must be left in the same state, so that the
            # membership draw that follows is identical too.
            np.random.seed(seed)
            sc.KFCM(n_clusters=4)._initialize_centers_kmeans_pp(X)
            after_opt = np.random.rand()
            np.random.seed(seed)
            scpp_original.KFCM(n_clusters=4)._initialize_centers_kmeans_pp(X)
            assert after_opt == np.random.rand()


def test_reference_package_is_not_importable_from_the_distribution():
    """The preserved originals must stay out of the production import path."""
    assert not (
        Path(sc.__file__).parent / "original"
    ).exists(), "reference implementations must not ship inside soft_clustering/"
    assert "scpp_original" not in sc.__file__
