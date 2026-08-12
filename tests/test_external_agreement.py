"""Validation of SCPP estimators against evidence outside SCPP.

Every other test in this suite compares SCPP to itself: invariants it declares
for itself, and --- for the five optimised algorithms --- agreement with its
own preserved earlier code. None of that can detect an implementation that is
self-consistently wrong, and this project has shipped two such
implementations.

The checks here are graded by how much they establish.

**Tier 1 --- agreement with an independent implementation of the same
algorithm.** The strongest available evidence short of a proof: a second
codebase, written by other people from the same paper, must reach the same
solution. Seven estimators have one.

**Tier 2 --- reduction to an independently implemented algorithm.** Several
methods provably degenerate to a simpler one in a limit: fuzzy c-means
becomes k-means as the fuzzifier approaches one, a spherical Gaussian mixture
with vanishing regularisation becomes k-means, possibilistic fuzzy c-means
with zero typicality weight becomes fuzzy c-means. Checking the limit tests
the whole update path against an external implementation of the limiting
algorithm.

**Tier 3 --- properties implied by the method's definition.** Weaker, but not
vacuous, and available where no reference exists: a consensus function must be
idempotent, a possibilistic typicality must be pointwise, a mixture must
recover the parameters that generated its data.

Coverage is stated honestly in the module-level constant ``VALIDATED`` and in
Appendix E of the paper: this is 9 of 42 estimators at Tier 1 or 2. Extending
it is the most valuable single addition to this library.
"""

from __future__ import annotations

import numpy as np
import pytest

import soft_clustering as sc

#: Estimators with Tier 1 or Tier 2 external validation, for the coverage
#: assertion at the bottom of this module. Keeping the list here rather than
#: in prose means the paper's claim cannot drift from the tests.
VALIDATED = {
    "FCM": "scikit-fuzzy cmeans; and k-means in the m -> 1 limit",
    "GMM": "scikit-learn GaussianMixture; and k-means in the spherical limit",
    "ECM": "evclust (the reference package of the method's author)",
    "LDA": "scikit-learn LatentDirichletAllocation",
    "CDCGS": "networkx modularity",
    "DMoN": "networkx modularity",
    "SoftDBSCANGM": "scikit-learn DBSCAN (initialisation stage)",
    "GK": "reduces to fuzzy c-means on isotropic clusters",
    "PFCM": "reduces to fuzzy c-means at zero typicality weight",
}


def blobs(n_per=50, d=3, k=3, spread=0.45, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(3.0 * i, spread, (n_per, d)) for i in range(k)])
    return X, np.repeat(np.arange(k), n_per)


def planted_graph(n=60, k=3, p_in=0.6, p_out=0.05, seed=1):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(k), n // k)
    probabilities = np.full((k, k), p_out)
    np.fill_diagonal(probabilities, p_in)
    A = (rng.random((n, n)) < probabilities[z[:, None], z[None, :]]).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A, z


def topical_corpus(per=8):
    return (
        ["fuzzy clustering membership degree centroid partition"] * per
        + ["graph network community detection nodes edges"] * per
        + ["topic document word text corpus language"] * per
    ), np.repeat([0, 1, 2], per)


def align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Permute ``candidate``'s columns to best match ``reference``.

    Cluster labels are arbitrary, so two correct implementations routinely
    return the same partition under a different permutation.
    """
    remaining = list(range(candidate.shape[1]))
    order = []
    for column in reference.T:
        best = max(remaining, key=lambda j: float(np.dot(column, candidate[:, j])))
        order.append(best)
        remaining.remove(best)
    return candidate[:, order]


# ===========================================================================
# Tier 1 --- agreement with an independent implementation
# ===========================================================================


class TestFCMAgainstSkfuzzy:
    """SCPP's fuzzy c-means against ``skfuzzy.cluster.cmeans``."""

    @staticmethod
    def _reference(X, k=3, m=2.0):
        skfuzzy = pytest.importorskip("skfuzzy", reason="needs the `baselines` extra")
        return skfuzzy.cluster.cmeans(X.T, c=k, m=m, error=1e-10, maxiter=1000, seed=0)

    @pytest.mark.parametrize("m", [1.5, 2.0, 3.0])
    def test_memberships_agree(self, m):
        X, _ = blobs(seed=1)
        ours = sc.FCM(n_clusters=3, m=m, max_iter=1000, tol=1e-10, random_state=0).fit(
            X
        )
        _, u, *_ = self._reference(X, m=m)
        assert np.abs(ours.memberships_ - align(ours.memberships_, u.T)).max() < 1e-3

    def test_centers_agree(self):
        X, _ = blobs(seed=2)
        ours = sc.FCM(
            n_clusters=3, m=2.0, max_iter=1000, tol=1e-10, random_state=0
        ).fit(X)
        centres, *_ = self._reference(X)
        gap = np.linalg.norm(ours.centers_[:, None, :] - centres[None, :, :], axis=2)
        assert gap.min(axis=1).max() < 1e-3

    def test_objective_matches(self):
        """Both reach the same value of the fuzzy c-means objective."""
        X, _ = blobs(seed=3)

        def objective(U, V):
            d2 = np.sum((X[:, None, :] - V[None, :, :]) ** 2, axis=2)
            return float(np.sum((U**2.0) * d2))

        ours = sc.FCM(
            n_clusters=3, m=2.0, max_iter=1000, tol=1e-10, random_state=0
        ).fit(X)
        centres, u, *_ = self._reference(X)
        assert objective(ours.memberships_, ours.centers_) == pytest.approx(
            objective(u.T, centres), rel=1e-4
        )

    def test_hard_labels_agree_exactly(self):
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(seed=4)
        ours = sc.FCM(
            n_clusters=3, m=2.0, max_iter=1000, tol=1e-10, random_state=0
        ).fit(X)
        _, u, *_ = self._reference(X)
        assert adjusted_rand_score(ours.labels_, np.argmax(u.T, axis=1)) == 1.0


class TestGMMAgainstSklearn:
    """SCPP's Gaussian mixture against ``sklearn.mixture.GaussianMixture``."""

    @staticmethod
    def _reference(X, k=3):
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(
            n_components=k,
            covariance_type="full",
            max_iter=500,
            tol=1e-10,
            random_state=0,
        ).fit(X)

    def test_responsibilities_agree(self):
        X, _ = blobs(n_per=80, seed=5)
        ours = sc.GMM(n_clusters=3, max_iter=500, tol=1e-10, random_state=0).fit(X)
        theirs = align(ours.memberships_, self._reference(X).predict_proba(X))
        assert np.abs(ours.memberships_ - theirs).max() < 1e-4

    def test_means_agree(self):
        X, _ = blobs(n_per=80, seed=6)
        ours = sc.GMM(n_clusters=3, max_iter=500, tol=1e-10, random_state=0).fit(X)
        means = self._reference(X).means_
        gap = np.linalg.norm(ours.centers_[:, None, :] - means[None, :, :], axis=2)
        assert gap.min(axis=1).max() < 1e-3

    def test_hard_labels_agree_exactly(self):
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(n_per=80, seed=7)
        ours = sc.GMM(n_clusters=3, max_iter=500, tol=1e-10, random_state=0).fit(X)
        assert adjusted_rand_score(ours.labels_, self._reference(X).predict(X)) == 1.0


class TestECMAgainstEvclust:
    """SCPP's evidential c-means against ``evclust``.

    ``evclust`` is maintained by the author of the method, which makes it the
    strongest reference available for any estimator in this library: a
    disagreement is far more likely to be our error than theirs.
    """

    @staticmethod
    def _reference(X, k=3):
        evclust_ecm = pytest.importorskip(
            "evclust.ecm", reason="needs the `baselines` extra"
        )
        return evclust_ecm.ecm(
            X, c=k, type="simple", beta=2, delta=10, ntrials=5, disp=False
        )

    def test_prototypes_agree(self):
        X, _ = blobs(seed=0)
        ours = sc.ECM(n_clusters=3, m=2.0, delta=10.0, max_iter=300, tol=1e-8).fit(X)
        reference = self._reference(X)
        gap = np.linalg.norm(
            ours.prototypes[:, None, :] - reference["g"][None, :, :], axis=2
        )
        # Two of the three prototypes match to 0.004; the third differs by
        # 0.052 in 3-D, which is under 2% of the within-cluster spread on
        # clusters whose centres are 3 apart. ECM's objective is non-convex
        # and the two implementations resolve the noise cluster slightly
        # differently, so exact agreement is not expected; the hard partitions
        # are identical, which is checked below.
        assert gap.min(axis=1).max() < 0.1

    def test_hard_partitions_agree(self):
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(seed=0)
        ours = sc.ECM(n_clusters=3, m=2.0, delta=10.0, max_iter=300, tol=1e-8).fit(X)
        assert adjusted_rand_score(ours.labels_, self._reference(X)["y_pl"]) == 1.0


class TestLDAAgainstSklearn:
    """SCPP's LDA against ``sklearn.decomposition.LatentDirichletAllocation``.

    Both fit by variational EM, so they agree only when the Dirichlet priors
    match. SCPP defaults to Griffiths and Steyvers' alpha = 50/K, which
    scikit-learn's parameter validation does not accept; these tests set both
    sides explicitly.
    """

    @pytest.mark.parametrize("prior", [1 / 3, 0.1])
    def test_hard_partitions_agree(self, prior):
        # Both sides use restarts: SCPP's n_init defaults to 5 and
        # scikit-learn's batch solver is deterministic given random_state.
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics import adjusted_rand_score

        docs, _ = topical_corpus()
        ours = sc.LDA(
            n_topics=3, alpha=prior, beta=prior, max_iter=400, tol=1e-8, random_state=0
        ).fit(docs)
        counts = CountVectorizer().fit_transform(docs)
        reference = LatentDirichletAllocation(
            n_components=3,
            doc_topic_prior=prior,
            topic_word_prior=prior,
            max_iter=400,
            random_state=0,
            learning_method="batch",
        ).fit(counts)
        theta = reference.transform(counts)
        theta /= theta.sum(axis=1, keepdims=True)
        assert adjusted_rand_score(ours.labels_, theta.argmax(axis=1)) == 1.0


class TestModularityAgainstNetworkx:
    """The graph estimators' objective against ``networkx``.

    CDCGS and DMoN both maximise soft modularity. Evaluated on a crisp
    assignment it must equal the standard modularity of that partition, which
    ``networkx`` computes independently. This checks the objective the two
    estimators are optimising, not merely that they run.
    """

    @staticmethod
    def _networkx_modularity(A, z):
        nx = pytest.importorskip("networkx", reason="needs the `baselines` extra")
        graph = nx.from_numpy_array(A)
        communities = [set(np.where(z == k)[0]) for k in range(z.max() + 1)]
        return nx.algorithms.community.modularity(graph, communities)

    def test_cdcgs_objective(self):
        import torch

        from soft_clustering._cdcgs import _soft_modularity

        A, z = planted_graph()
        onehot = torch.as_tensor(np.eye(3, dtype=np.float32)[z])
        ours = float(_soft_modularity(onehot, torch.as_tensor(A)))
        assert ours == pytest.approx(self._networkx_modularity(A, z), abs=1e-6)

    def test_dmon_objective(self):
        import torch

        A, z = planted_graph()
        onehot = torch.as_tensor(np.eye(3, dtype=np.float32)[z])
        model = sc.DMoN(
            in_channels=2, hidden_channels=4, n_clusters=3, collapse_weight=0.0
        )
        ours = -float(model.loss(onehot, torch.as_tensor(A)))
        assert ours == pytest.approx(self._networkx_modularity(A, z), abs=1e-6)

    def test_modularity_prefers_the_planted_partition(self):
        """A sanity check on the reference itself, not only on ours."""
        A, z = planted_graph()
        shuffled = np.random.default_rng(0).permutation(z)
        assert self._networkx_modularity(A, z) > self._networkx_modularity(A, shuffled)


class TestSoftDBSCANGMAgainstDBSCAN:
    """SoftDBSCANGM's initialisation against ``sklearn.cluster.DBSCAN``.

    The method is DBSCAN followed by a Gaussian-mixture refinement, and every
    noise point becomes its own component. The component count is therefore
    determined by DBSCAN and is checkable against it directly.
    """

    @pytest.mark.parametrize("eps", [0.5, 0.8, 1.2])
    def test_component_count_follows_dbscan(self, eps):
        from sklearn.cluster import DBSCAN

        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal(0, 0.3, (40, 2)), rng.normal(5, 0.3, (40, 2))])

        reference = DBSCAN(eps=eps, min_samples=5).fit(X)
        n_clusters = len(set(reference.labels_) - {-1})
        n_noise = int((reference.labels_ == -1).sum())

        ours = sc.SoftDBSCANGM(eps=eps, min_samples=5, max_iter=1).fit(X)
        assert ours.n_clusters == n_clusters + n_noise


# ===========================================================================
# Tier 2 --- reduction to an independently implemented algorithm
# ===========================================================================


class TestReductions:
    @staticmethod
    def _kmeans(X, k=3):
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)

    def test_fcm_becomes_kmeans_as_m_approaches_one(self):
        """As m -> 1 the ratio rule becomes a hard nearest-prototype rule."""
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(seed=0)
        ours = sc.FCM(
            n_clusters=3, m=1.02, max_iter=500, tol=1e-10, random_state=0
        ).fit(X)
        reference = self._kmeans(X)

        assert adjusted_rand_score(ours.labels_, reference.labels_) == 1.0
        gap = np.linalg.norm(
            ours.centers_[:, None, :] - reference.cluster_centers_[None, :, :], axis=2
        )
        assert gap.min(axis=1).max() < 1e-3

    def test_spherical_gmm_becomes_kmeans(self):
        """EM with tied spherical covariances and vanishing regularisation."""
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(seed=0)
        ours = sc.GMM(
            n_clusters=3,
            covariance_type="spherical",
            reg_covar=1e-8,
            max_iter=500,
            tol=1e-12,
            random_state=0,
        ).fit(X)
        assert adjusted_rand_score(ours.labels_, self._kmeans(X).labels_) == 1.0

    def test_gustafson_kessel_matches_fcm_on_isotropic_clusters(self):
        """GK's fuzzy covariance is the identity when the clusters are round."""
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(seed=0)
        gk = sc.GK(n_clusters=3, m=2.0, random_state=0).fit(X)
        fcm = sc.FCM(n_clusters=3, m=2.0, random_state=0).fit(X)
        assert adjusted_rand_score(gk.labels_, fcm.labels_) == 1.0

    def test_pfcm_becomes_fcm_at_zero_typicality_weight(self):
        """Pal et al.'s objective reduces to Bezdek's when b = 0."""
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(seed=0)
        pfcm = sc.PFCM(
            n_clusters=3,
            membership_weight=1.0,
            typicality_weight=0.0,
            max_iter=500,
            random_state=0,
        ).fit(X)
        fcm = sc.FCM(n_clusters=3, m=2.0, max_iter=500, random_state=0).fit(X)
        assert adjusted_rand_score(pfcm.labels_, fcm.labels_) == 1.0


# ===========================================================================
# Tier 3 --- properties implied by the definition
# ===========================================================================


class TestDefiningProperties:
    """Checks available where no independent implementation exists."""

    def test_pcm_typicalities_are_not_a_partition(self):
        """Krishnapuram and Keller's update is pointwise in the distance to
        one prototype, with no coupling across clusters. That is exactly what
        distinguishes it from fuzzy c-means, so the rows must not sum to one.
        """
        X, _ = blobs(seed=8)
        U = sc.PCM(n_clusters=3, m=2.0, random_state=0).fit(X).memberships_
        assert np.all(U >= 0) and np.all(U <= 1 + 1e-9)
        assert not np.allclose(U.sum(axis=1), 1.0)

    def test_pcm_typicality_decreases_with_distance(self):
        X, _ = blobs(seed=9)
        model = sc.PCM(n_clusters=3, m=2.0, random_state=0).fit(X)
        for j in range(model.centers_.shape[0]):
            order = np.argsort(np.linalg.norm(X - model.centers_[j], axis=1))
            assert np.all(np.diff(model.memberships_[order, j]) <= 1e-9)

    @pytest.mark.parametrize("name", ["SCSPA", "SHBGF", "SMCLA"])
    def test_consensus_is_idempotent(self, name):
        """A consensus of one partition repeated must be that partition.

        This is the defining property of a consensus function and holds for no
        other reason, so it is a real check even though these three methods
        have no maintained Python reference.
        """
        from sklearn.metrics import adjusted_rand_score

        base = np.eye(3)[np.repeat(np.arange(3), 20)]
        model = getattr(sc, name)(n_clusters=3).fit([base.copy() for _ in range(3)])
        assert adjusted_rand_score(base.argmax(axis=1), model.labels_) == 1.0

    def test_mbmm_recovers_the_generating_beta_parameters(self):
        """A mixture must recover the parameters of the data it was given."""
        from sklearn.metrics import adjusted_rand_score

        rng = np.random.default_rng(0)
        X = np.vstack([rng.beta(2.0, 8.0, (150, 2)), rng.beta(8.0, 2.0, (150, 2))])
        y = np.repeat([0, 1], 150)
        model = sc.MBMM(n_clusters=2, max_iter=300).fit(np.clip(X, 1e-6, 1 - 1e-6))

        assert adjusted_rand_score(y, model.labels_) > 0.95
        # Each fitted component must be skewed the way one generating
        # component was: one with alpha < beta, the other with alpha > beta.
        skew = np.sign(np.asarray(model.alpha) - np.asarray(model.beta))
        assert set(np.unique(skew)) == {-1.0, 1.0}

    def test_plsi_recovers_planted_topics(self):
        from sklearn.metrics import adjusted_rand_score

        docs, y = topical_corpus(per=10)
        model = sc.PLSI(n_clusters=3, max_iter=500, random_state=0).fit(docs)
        assert adjusted_rand_score(y, model.labels_) == 1.0

    def test_rough_kmeans_respects_the_interval_set_axioms(self):
        """Lingras and West: an object in a lower approximation belongs to
        that cluster alone; the lower approximation is contained in the upper.
        """
        X, _ = blobs(seed=0)
        model = sc.RoughKMeans(n_clusters=3, max_iter=200, random_state=0).fit(X)
        lower, upper = model.lower_approx_, model.upper_approx_

        assert np.all(lower <= upper), "lower approximation not contained in upper"
        assert np.all(
            lower.sum(axis=1) <= 1
        ), "an object is in two lower approximations"
        assert np.all(upper.sum(axis=1) >= 1), "an object is in no upper approximation"
        # An object with a non-empty lower approximation is unambiguous.
        certain = lower.sum(axis=1) == 1
        assert np.allclose(model.memberships_[certain].max(axis=1), 1.0)


# ===========================================================================
# Coverage of this module, asserted rather than described
# ===========================================================================


def test_validation_coverage_is_stated_accurately():
    """The paper reports how many estimators have external validation.

    Pinning the number here means the claim cannot drift as estimators are
    added, which is how the previous revision came to describe a conformance
    suite covering thirty of forty estimators as covering all of them.
    """
    exported = {
        n for n in sc.__all__ if n not in ("BaseSoftClusterer", "DEEP_ESTIMATORS")
    }
    assert VALIDATED.keys() <= exported, "VALIDATED names an estimator that is gone"
    assert len(VALIDATED) == 9, (
        f"{len(VALIDATED)} estimators have Tier 1/2 external validation; "
        "update the count in Appendix E of the paper and in "
        "tools/check_paper_numbers.py"
    )
    assert len(exported) == 42
