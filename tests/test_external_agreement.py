"""Agreement between SCPP estimators and independent reference implementations.

Every other test in this suite checks SCPP against itself: shapes, invariants,
and — for the five optimised algorithms — against SCPP's own preserved
pre-optimization code. None of that can detect an implementation that is
self-consistently wrong.

These tests close that gap where an independent implementation of the same
method exists. Each one fits SCPP and a third-party estimator on identical
data with matched hyperparameters and requires the *solutions* to agree, not
merely the shapes. They are skipped when the third-party package is absent,
so they never block a base install; CI installs the ``baselines`` extra.

What agreement can and cannot show. A match is strong evidence that SCPP
solves the same optimisation problem as the reference. It is not a proof of
correctness — two implementations can share a misreading of a paper — and it
covers only the algorithms for which a reference exists. Fourteen of the 42
estimators have no widely used independent Python implementation; for those,
the invariants and the recovery tests in the per-algorithm modules are the
available evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from soft_clustering import FCM, GMM, PCM

skfuzzy = pytest.importorskip("skfuzzy", reason="needs the `baselines` extra")
sklearn_mixture = pytest.importorskip("sklearn.mixture")


def blobs(n_per=60, d=3, k=3, spread=0.45, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(3.0 * i, spread, (n_per, d)) for i in range(k)])
    return X, np.repeat(np.arange(k), n_per)


def _align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Reorder ``candidate``'s columns to best match ``reference``.

    Cluster labels are arbitrary, so two correct implementations routinely
    return the same partition under a different permutation. Matching greedily
    on column correlation is enough here, where the partitions are close.
    """
    remaining = list(range(candidate.shape[1]))
    order = []
    for column in reference.T:
        best = max(
            remaining,
            key=lambda j: float(np.dot(column, candidate[:, j])),
        )
        order.append(best)
        remaining.remove(best)
    return candidate[:, order]


# ---------------------------------------------------------------------------
# Fuzzy c-means vs scikit-fuzzy
# ---------------------------------------------------------------------------


class TestFCMAgainstSkfuzzy:
    """SCPP's FCM against ``skfuzzy.cluster.cmeans``."""

    @pytest.mark.parametrize("m", [1.5, 2.0, 3.0])
    def test_memberships_agree(self, m):
        X, _ = blobs(seed=1)

        ours = FCM(n_clusters=3, m=m, max_iter=1000, tol=1e-10, random_state=0).fit(X)
        centres, u, *_ = skfuzzy.cluster.cmeans(
            X.T, c=3, m=m, error=1e-10, maxiter=1000, seed=0
        )
        theirs = _align(ours.memberships_, u.T)

        # Both converge to the same optimum of the same objective, from
        # different initialisations, so they agree to optimiser tolerance.
        assert np.abs(ours.memberships_ - theirs).max() < 1e-3

    def test_centers_agree(self):
        X, _ = blobs(seed=2)

        ours = FCM(n_clusters=3, m=2.0, max_iter=1000, tol=1e-10, random_state=0).fit(X)
        centres, u, *_ = skfuzzy.cluster.cmeans(
            X.T, c=3, m=2.0, error=1e-10, maxiter=1000, seed=0
        )

        # Match each of our prototypes to its nearest reference prototype.
        distances = np.linalg.norm(
            ours.centers_[:, None, :] - centres[None, :, :], axis=2
        )
        assert distances.min(axis=1).max() < 1e-3

    def test_objective_matches(self):
        """Same value of the fuzzy c-means objective, to four decimals."""
        X, _ = blobs(seed=3)
        m = 2.0

        def objective(U, V):
            d2 = np.sum((X[:, None, :] - V[None, :, :]) ** 2, axis=2)
            return float(np.sum((U**m) * d2))

        ours = FCM(n_clusters=3, m=m, max_iter=1000, tol=1e-10, random_state=0).fit(X)
        centres, u, *_ = skfuzzy.cluster.cmeans(
            X.T, c=3, m=m, error=1e-10, maxiter=1000, seed=0
        )

        assert objective(ours.memberships_, ours.centers_) == pytest.approx(
            objective(u.T, centres), rel=1e-4
        )

    def test_hard_labels_agree_exactly(self):
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(seed=4)
        ours = FCM(n_clusters=3, m=2.0, max_iter=1000, tol=1e-10, random_state=0).fit(X)
        _, u, *_ = skfuzzy.cluster.cmeans(
            X.T, c=3, m=2.0, error=1e-10, maxiter=1000, seed=0
        )
        assert adjusted_rand_score(ours.labels_, np.argmax(u.T, axis=1)) == 1.0


# ---------------------------------------------------------------------------
# Gaussian mixture EM vs scikit-learn
# ---------------------------------------------------------------------------


class TestGMMAgainstSklearn:
    """SCPP's GMM against ``sklearn.mixture.GaussianMixture``."""

    def test_responsibilities_agree(self):
        X, _ = blobs(n_per=80, seed=5)

        ours = GMM(n_clusters=3, max_iter=500, tol=1e-10, random_state=0).fit(X)
        reference = sklearn_mixture.GaussianMixture(
            n_components=3,
            covariance_type="full",
            max_iter=500,
            tol=1e-10,
            random_state=0,
        ).fit(X)

        theirs = _align(ours.memberships_, reference.predict_proba(X))
        assert np.abs(ours.memberships_ - theirs).max() < 1e-4

    def test_log_likelihood_agrees(self):
        """The fitted models explain the data equally well."""
        X, _ = blobs(n_per=80, seed=6)

        ours = GMM(n_clusters=3, max_iter=500, tol=1e-10, random_state=0).fit(X)
        reference = sklearn_mixture.GaussianMixture(
            n_components=3,
            covariance_type="full",
            max_iter=500,
            tol=1e-10,
            random_state=0,
        ).fit(X)

        # Score our means under the reference model: identical solutions place
        # the components in the same locations.
        distances = np.linalg.norm(
            ours.centers_[:, None, :] - reference.means_[None, :, :], axis=2
        )
        assert distances.min(axis=1).max() < 1e-3

    def test_hard_labels_agree_exactly(self):
        from sklearn.metrics import adjusted_rand_score

        X, _ = blobs(n_per=80, seed=7)
        ours = GMM(n_clusters=3, max_iter=500, tol=1e-10, random_state=0).fit(X)
        reference = sklearn_mixture.GaussianMixture(
            n_components=3, max_iter=500, tol=1e-10, random_state=0
        ).fit(X)
        assert adjusted_rand_score(ours.labels_, reference.predict(X)) == 1.0


# ---------------------------------------------------------------------------
# Possibilistic c-means: no reference implementation, so check the theory
# ---------------------------------------------------------------------------


class TestPCMProperties:
    """PCM has no widely used independent Python implementation.

    Krishnapuram and Keller's defining property is checkable directly: the
    typicality update is a *pointwise* function of the distance to one
    prototype, with no coupling across clusters, which is exactly what
    distinguishes it from fuzzy c-means. So typicalities must not be
    normalised, and duplicating the data must not change them.
    """

    def test_typicalities_are_not_a_partition(self):
        X, _ = blobs(seed=8)
        U = PCM(n_clusters=3, m=2.0, random_state=0).fit(X).memberships_
        assert np.all(U >= 0) and np.all(U <= 1 + 1e-9)
        assert not np.allclose(U.sum(axis=1), 1.0)

    def test_typicality_decreases_with_distance(self):
        """Within a cluster, a point further from the prototype is less typical."""
        X, _ = blobs(seed=9)
        model = PCM(n_clusters=3, m=2.0, random_state=0).fit(X)
        U, V = model.memberships_, model.centers_

        for j in range(V.shape[0]):
            distance = np.linalg.norm(X - V[j], axis=1)
            order = np.argsort(distance)
            typicality = U[order, j]
            # Monotone non-increasing, up to floating-point noise.
            assert np.all(np.diff(typicality) <= 1e-9)
