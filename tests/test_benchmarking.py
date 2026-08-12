"""Tests for ``soft_clustering.benchmarking``.

The benchmarking suite ships inside the installed package, so it falls under
the same coverage source as the estimators and needs the same level of test.
These tests exercise the adapter's interface detection, the validity metrics,
the dataset registry, the four benchmark backends, and — importantly — the
optional-dependency guards that keep the module importable on a bare install.

The adapter is tested against purpose-built fakes rather than real estimators,
so that each of the three fit interfaces and each membership storage layout is
covered deterministically; one end-to-end test with a real estimator pins the
integration.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import scipy.sparse as sp

from soft_clustering.benchmarking import _optional, datasets, metrics

# --------------------------------------------------------------------------
# Optional third-party dependencies
# --------------------------------------------------------------------------

try:
    import pandas  # noqa: F401

    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

try:
    import psutil  # noqa: F401

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    import sklearn  # noqa: F401

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

try:
    import tabulate  # noqa: F401

    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False

needs_pandas = pytest.mark.skipif(not HAVE_PANDAS, reason="pandas not installed")
needs_psutil = pytest.mark.skipif(not HAVE_PSUTIL, reason="psutil not installed")
needs_sklearn = pytest.mark.skipif(
    not HAVE_SKLEARN, reason="scikit-learn not installed"
)
needs_tabulate = pytest.mark.skipif(not HAVE_TABULATE, reason="tabulate not installed")


# --------------------------------------------------------------------------
# Fakes, one per fit interface the adapter must detect
# --------------------------------------------------------------------------


def _random_memberships(n, k, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.random((n, k))
    return U / U.sum(axis=1, keepdims=True)


class FitPredictWithK:
    """Interface 1: ``fit_predict(X, K)`` — K required at call time."""

    def __init__(self):
        self.memberships_ = None
        self.centers_ = None

    def fit_predict(self, X, K):
        self.memberships_ = _random_memberships(X.shape[0], K)
        self.centers_ = np.zeros((K, X.shape[1]))
        return self.memberships_


class FitPredictNoK:
    """Interface 2: ``fit_predict(X)`` — K came from ``__init__``.

    Returns ``(labels, U)`` and stores centres under ``centroids``, mirroring
    CAFCM.
    """

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.centroids = None

    def fit_predict(self, X):
        U = _random_memberships(X.shape[0], self.n_clusters, seed=1)
        self.centroids = np.zeros((self.n_clusters, X.shape[1]))
        return np.argmax(U, axis=1), U


class FitOnly:
    """Interface 3: sklearn-style ``fit(X)``.

    Stores the membership matrix transposed as (K, n) under ``U`` and centres
    under ``V``, mirroring KFCM.
    """

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.U = None
        self.V = None

    def fit(self, X):
        self.U = _random_memberships(X.shape[0], self.n_clusters, seed=2).T
        self.V = np.zeros((self.n_clusters, X.shape[1]))
        return self


class NoInterface:
    """Exposes neither ``fit`` nor ``fit_predict``."""


class FitNoMembership:
    """Fits, but exposes no membership matrix and no labels."""

    def fit(self, X):
        return self


class PredictRaises:
    """``predict`` fails, so consumers must fall back to ``labels_``."""

    def fit(self, X):
        self.labels_ = np.arange(X.shape[0]) % 2
        return self

    def predict(self, X):
        raise RuntimeError("predict is unavailable for this model")


class LabelsOnly:
    """No ``predict``; hard labels only."""

    def fit(self, X):
        self.labels_ = np.arange(X.shape[0]) % 2
        return self


class MembershipOnly:
    """Neither ``predict`` nor ``labels_``; labels must come from argmax."""

    def fit(self, X):
        self.U = _random_memberships(X.shape[0], 2, seed=3)
        return self


class MeansModel:
    """Stores centres under ``means_`` and memberships under
    ``responsibilities_``, mirroring GMM."""

    def fit(self, X):
        self.responsibilities_ = _random_memberships(X.shape[0], 2, seed=4)
        self.means_ = np.zeros((2, X.shape[1]))
        return self


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return np.vstack(
        [rng.normal([0, 0], 0.3, (15, 2)), rng.normal([4, 4], 0.3, (15, 2))]
    )


# ==========================================================================
# Package surface
# ==========================================================================


class TestPackageSurface:
    def test_all_exports_are_importable(self):
        import soft_clustering.benchmarking as B

        for name in B.__all__:
            assert getattr(B, name) is not None

    def test_dir_includes_lazy_exports(self):
        import soft_clustering.benchmarking as B

        assert set(B.__all__).issubset(set(dir(B)))

    def test_unknown_attribute_raises(self):
        import soft_clustering.benchmarking as B

        with pytest.raises(AttributeError, match="no attribute"):
            B.NoSuchBenchmark

    def test_submodules_are_importable(self):
        """``from soft_clustering.benchmarking import metrics`` must work even
        though the lazy ``__getattr__`` does not list submodules."""
        from soft_clustering.benchmarking import adapter, base, report, runner

        assert all(m is not None for m in (adapter, base, report, runner))


class TestAttributeRegistryIsShared:
    """The benchmarking code must not keep its own copy of the attribute names
    the estimator protocol defines. Two copies drifted apart once already: this
    module knew five membership attributes while ``_base`` knew ten."""

    def test_adapter_uses_the_protocol_registry(self):
        from soft_clustering._base import BaseSoftClusterer
        from soft_clustering.benchmarking import adapter

        assert adapter._MEMBERSHIP_ATTRS is BaseSoftClusterer._membership_attrs
        assert adapter._CENTER_ATTRS is BaseSoftClusterer._centers_attrs

    def test_quality_backend_covers_the_protocol_registry(self):
        from soft_clustering._base import BaseSoftClusterer
        from soft_clustering.benchmarking.benchmark import clustering_quality

        assert set(BaseSoftClusterer._membership_attrs).issubset(
            clustering_quality._MEMBERSHIP_ATTRS
        )

    def test_every_estimator_membership_attr_is_known_to_the_benchmarks(self):
        """Guards the direction that actually bites: an estimator storing its
        memberships under a name the benchmarks do not search."""
        from soft_clustering._base import BaseSoftClusterer
        from soft_clustering.benchmarking import adapter

        for attr in BaseSoftClusterer._membership_attrs:
            assert attr in adapter._MEMBERSHIP_ATTRS


# ==========================================================================
# Optional-dependency guards
# ==========================================================================


class TestOptionalDependencies:
    def test_require_pandas_raises_when_absent(self, monkeypatch):
        monkeypatch.setattr(_optional, "pd", None)
        with pytest.raises(ImportError, match=r"soft-clustering\[bench\]"):
            _optional.require_pandas("Feature()")

    def test_require_psutil_raises_when_absent(self, monkeypatch):
        monkeypatch.setattr(_optional, "psutil", None)
        with pytest.raises(ImportError, match=r"soft-clustering\[bench\]"):
            _optional.require_psutil("Feature()")

    def test_error_names_the_feature_and_package(self, monkeypatch):
        monkeypatch.setattr(_optional, "pd", None)
        with pytest.raises(ImportError) as exc:
            _optional.require_pandas("ClusteringBenchmark.run()")
        assert "ClusteringBenchmark.run()" in str(exc.value)
        assert "pandas" in str(exc.value)

    @needs_pandas
    def test_require_pandas_returns_module_when_present(self):
        assert _optional.require_pandas("Feature()") is not None

    @needs_psutil
    def test_require_psutil_returns_module_when_present(self):
        assert _optional.require_psutil("Feature()") is not None

    def test_runner_reports_missing_pandas(self, monkeypatch, X):
        from soft_clustering.benchmarking import ClusteringBenchmark

        monkeypatch.setattr(_optional, "pd", None)
        with pytest.raises(ImportError, match=r"soft-clustering\[bench\]"):
            ClusteringBenchmark(models=[], benchmarks=[]).run(X)

    def test_memory_backend_reports_missing_psutil(self, monkeypatch):
        from soft_clustering.benchmarking import MemoryBenchmark

        monkeypatch.setattr(_optional, "psutil", None)
        with pytest.raises(ImportError, match=r"soft-clustering\[bench\]"):
            MemoryBenchmark()._memory_mb()


# ==========================================================================
# BenchmarkAdapter
# ==========================================================================


class TestAdapterDispatch:
    def test_fit_predict_with_required_k(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(FitPredictWithK(), n_clusters=3).fit(X)
        assert a.membership_.shape == (30, 3)
        assert a.centers_.shape == (3, 2)
        assert a.labels_.shape == (30,)

    def test_fit_predict_with_required_k_missing_n_clusters(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        with pytest.raises(ValueError, match="requires K"):
            BenchmarkAdapter(FitPredictWithK()).fit(X)

    def test_fit_predict_without_k(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(FitPredictNoK(n_clusters=3)).fit(X)
        assert a.membership_.shape == (30, 3)
        assert a.centers_.shape == (3, 2)

    def test_sklearn_style_fit_with_transposed_memberships(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(FitOnly(n_clusters=2)).fit(X)
        # stored (2, 30) on the model; the adapter must hand back (30, 2)
        assert a.model.U.shape == (2, 30)
        assert a.membership_.shape == (30, 2)
        assert a.centers_.shape == (2, 2)

    def test_model_without_any_fit_interface(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        with pytest.raises(TypeError, match="neither"):
            BenchmarkAdapter(NoInterface()).fit(X)

    def test_sparse_membership_is_densified(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        class SparseModel:
            def fit_predict(self, X):
                return sp.csr_matrix(_random_memberships(X.shape[0], 3))

        a = BenchmarkAdapter(SparseModel()).fit(X)
        assert isinstance(a.membership_, np.ndarray)
        assert a.membership_.shape == (30, 3)

    def test_tuple_return_with_labels_only(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        class LabelsOnly:
            def fit_predict(self, X):
                return (np.zeros(X.shape[0], dtype=int),)

        a = BenchmarkAdapter(LabelsOnly()).fit(X)
        assert a.membership_ is None
        assert a.labels_.shape == (30,)

    def test_no_membership_leaves_attributes_none(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(FitNoMembership()).fit(X)
        assert a.membership_ is None
        assert a.centers_ is None

    def test_centers_found_under_alternative_attribute(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(MeansModel()).fit(X)
        assert a.membership_.shape == (30, 2)
        assert a.centers_.shape == (2, 2)  # located via means_

    def test_labels_derived_from_membership_argmax(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(FitPredictWithK(), n_clusters=3).fit(X)
        assert np.array_equal(a.labels_, np.argmax(a.membership_, axis=1))

    def test_predict_before_fit_raises(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter

        with pytest.raises(RuntimeError, match="before predict"):
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2).predict(X)


class TestAdapterIdentity:
    """Regression tests for the removal of the ``__class__`` property."""

    def test_isinstance_reports_the_adapter(self):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(FitOnly())
        assert isinstance(a, BenchmarkAdapter)
        assert type(a) is BenchmarkAdapter

    def test_wrapped_class_is_not_spoofed(self):
        from soft_clustering.benchmarking import BenchmarkAdapter

        assert not isinstance(BenchmarkAdapter(FitOnly()), FitOnly)

    def test_name_defaults_to_wrapped_class(self):
        from soft_clustering.benchmarking import BenchmarkAdapter

        assert BenchmarkAdapter(FitOnly()).name == "FitOnly"

    def test_name_can_be_overridden(self):
        from soft_clustering.benchmarking import BenchmarkAdapter

        assert BenchmarkAdapter(FitOnly(), name="KFCM").name == "KFCM"

    def test_model_name_helper(self):
        from soft_clustering.benchmarking import BenchmarkAdapter
        from soft_clustering.benchmarking.base import model_name

        assert model_name(BenchmarkAdapter(FitOnly())) == "FitOnly"
        assert model_name(BenchmarkAdapter(FitOnly(), name="KFCM")) == "KFCM"
        assert model_name(FitOnly()) == "FitOnly"

    def test_model_name_ignores_non_string_name(self):
        from soft_clustering.benchmarking.base import model_name

        class Weird:
            name = 42

        assert model_name(Weird()) == "Weird"

    def test_attribute_delegation(self):
        from soft_clustering.benchmarking import BenchmarkAdapter

        a = BenchmarkAdapter(FitOnly(n_clusters=5))
        assert a.n_clusters is None  # the adapter's own, not the model's
        assert a.model.n_clusters == 5

    def test_delegation_reaches_wrapped_attributes(self):
        from soft_clustering.benchmarking import BenchmarkAdapter

        model = FitOnly()
        model.custom_attr = "hello"
        assert BenchmarkAdapter(model).custom_attr == "hello"

    def test_missing_attribute_still_raises(self):
        from soft_clustering.benchmarking import BenchmarkAdapter

        with pytest.raises(AttributeError):
            BenchmarkAdapter(FitOnly()).definitely_not_here


# ==========================================================================
# Metrics
# ==========================================================================


_ALL_SOFT_KEYS = {
    "partition_coefficient",
    "modified_partition_coefficient",
    "partition_entropy",
    "fuzzy_hypervolume",
    "xie_beni",
    "fuzzy_compactness",
    "fuzzy_separation",
}


class TestSoftMetrics:
    def test_crisp_partition_is_maximally_certain(self):
        U = np.eye(3)[np.array([0, 1, 2, 0, 1, 2])]
        assert metrics.partition_coefficient(U) == pytest.approx(1.0)
        assert metrics.partition_entropy(U) == pytest.approx(0.0, abs=1e-9)
        assert metrics.modified_partition_coefficient(U) == pytest.approx(1.0)

    def test_uniform_partition_is_maximally_uncertain(self):
        c = 4
        U = np.full((10, c), 1.0 / c)
        assert metrics.partition_coefficient(U) == pytest.approx(1.0 / c)
        assert metrics.modified_partition_coefficient(U) == pytest.approx(0.0)
        assert metrics.partition_entropy(U) == pytest.approx(np.log(c), abs=1e-6)

    def test_fuzzy_hypervolume_matches_the_analytic_volume(self):
        """FHV = sum_i sqrt(det F_i) with F_i the fuzzy covariance.

        For two crisp isotropic clusters of standard deviation s in d
        dimensions, det F_i = s^(2d), so the index converges to 2 * s^d.
        """
        rng = np.random.default_rng(0)
        s, d, n = 0.5, 2, 4000
        X = np.vstack([rng.normal(0, s, (n, d)), rng.normal(10, s, (n, d))])
        U = np.zeros((2 * n, 2))
        U[:n, 0] = 1.0
        U[n:, 1] = 1.0
        centers = np.vstack([X[:n].mean(axis=0), X[n:].mean(axis=0)])

        assert metrics.fuzzy_hypervolume(X, U, centers, m=1.0) == pytest.approx(
            2 * s**d, rel=0.02
        )

    def test_fuzzy_hypervolume_scales_as_a_volume(self):
        """Scaling the data by a must scale a d-dimensional volume by a**d."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 3))
        U = np.zeros((500, 2))
        U[:250, 0] = 1.0
        U[250:, 1] = 1.0
        centers = np.vstack([X[:250].mean(axis=0), X[250:].mean(axis=0)])

        base = metrics.fuzzy_hypervolume(X, U, centers, m=1.0)
        scaled = metrics.fuzzy_hypervolume(2 * X, U, 2 * centers, m=1.0)
        assert scaled / base == pytest.approx(2**3, rel=1e-6)

    def test_fuzzy_hypervolume_is_zero_for_a_degenerate_cluster(self):
        """A cluster confined to a line occupies no volume."""
        X = np.zeros((100, 2))
        X[:, 0] = np.linspace(0, 1, 100)
        U = np.ones((100, 1))
        centers = X.mean(axis=0, keepdims=True)
        assert metrics.fuzzy_hypervolume(X, U, centers, m=1.0) == pytest.approx(0.0)

    def test_fuzzy_separation_is_mean_pairwise_distance(self):
        centers = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert metrics.fuzzy_separation_index(centers) == pytest.approx(5.0)

    def test_xie_beni_is_finite_and_positive(self):
        X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
        U = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        centers = np.array([[0.05, 0.05], [5.05, 5.05]])
        xb = metrics.xie_beni_index(X, U, centers)
        assert np.isfinite(xb) and xb > 0

    @pytest.mark.parametrize("separation", [2.0, 10.0, 50.0])
    def test_xie_beni_falls_as_clusters_separate(self, separation):
        """Two crisp clusters of fixed internal spread. Holding compactness
        constant, the index must fall as the centres move apart: the numerator
        is unchanged and the denominator grows with the centre distance."""
        d = separation
        X = np.array([[-0.1, 0.0], [0.1, 0.0], [d - 0.1, 0.0], [d + 0.1, 0.0]])
        U = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        centers = np.array([[0.0, 0.0], [d, 0.0]])

        # numerator = 4 * 0.1**2 = 0.04, denominator = n * d**2 = 4 * d**2
        assert metrics.xie_beni_index(X, U, centers) == pytest.approx(0.04 / (4 * d**2))

    def test_fuzzy_compactness_is_non_negative(self):
        X = np.zeros((4, 2))
        U = np.full((4, 2), 0.5)
        assert metrics.fuzzy_compactness(X, U, np.zeros((2, 2))) == pytest.approx(0.0)

    def test_soft_clustering_metrics_without_centers(self):
        """Every key is present; the prototype-based ones are nan."""
        U = _random_memberships(20, 3)
        out = metrics.soft_clustering_metrics(np.zeros((20, 2)), U)
        assert set(out) == _ALL_SOFT_KEYS
        assert np.isfinite(out["partition_coefficient"])
        for key in (
            "fuzzy_hypervolume",
            "xie_beni",
            "fuzzy_compactness",
            "fuzzy_separation",
        ):
            assert np.isnan(out[key])

    def test_soft_clustering_metrics_with_centers(self):
        rng = np.random.default_rng(0)
        U = _random_memberships(20, 3)
        X = rng.normal(size=(20, 2))
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        out = metrics.soft_clustering_metrics(X, U, centers=centers)
        assert set(out) == _ALL_SOFT_KEYS
        assert all(np.isfinite(v) for v in out.values())

    def test_xie_beni_diverges_for_coincident_prototypes(self):
        """Zero separation means the index is unbounded, not zero.

        The previous implementation replaced the zero separation with inf and
        then divided by it, silently reporting a perfect 0.0 for a completely
        degenerate solution.
        """
        X = np.random.default_rng(0).normal(size=(10, 2))
        U = _random_memberships(10, 3)
        assert metrics.xie_beni_index(X, U, np.zeros((3, 2))) == np.inf

    def test_fuzziness_indices_are_nan_for_unnormalised_memberships(self):
        """PC, MPC and PE are only defined under the partition constraint.

        Reporting them for possibilistic typicalities puts numbers on
        different scales into the same table column.
        """
        rng = np.random.default_rng(0)
        typicalities = np.abs(rng.normal(size=(20, 3)))
        X = rng.normal(size=(20, 2))
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        out = metrics.soft_clustering_metrics(X, typicalities, centers)
        for key in (
            "partition_coefficient",
            "modified_partition_coefficient",
            "partition_entropy",
        ):
            assert np.isnan(out[key])
        # The prototype-based indices do not need normalisation.
        assert np.isfinite(out["xie_beni"])

    def test_declared_constraint_overrides_detection(self):
        U = _random_memberships(20, 3)
        out = metrics.soft_clustering_metrics(
            np.zeros((20, 2)), U, partition_constrained=False
        )
        assert np.isnan(out["partition_coefficient"])

    def test_is_partition_constrained(self):
        assert metrics.is_partition_constrained(_random_memberships(10, 3))
        assert not metrics.is_partition_constrained(np.ones((10, 3)))


@needs_sklearn
class TestHardMetrics:
    def test_clustering_metrics_unsupervised(self):
        X = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 5])
        labels = np.array([0] * 5 + [1] * 5)
        out = metrics.clustering_metrics(X, labels)
        assert {"silhouette", "calinski_harabasz", "davies_bouldin"} <= set(out)

    def test_clustering_metrics_supervised(self):
        X = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 5])
        labels = np.array([0] * 5 + [1] * 5)
        out = metrics.clustering_metrics(X, labels, y_true=labels)
        assert out["ari"] == pytest.approx(1.0)
        assert out["nmi"] == pytest.approx(1.0)

    def test_single_cluster_reports_internal_metrics_as_nan(self):
        """Keys are always present so benchmark tables have no ragged rows."""
        X = np.random.default_rng(0).normal(size=(10, 2))
        out = metrics.clustering_metrics(X, np.zeros(10, dtype=int))
        assert set(out) == {
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "ari",
            "nmi",
        }
        assert np.isnan(out["silhouette"])

    def test_external_metrics_are_nan_without_ground_truth(self):
        X = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 5])
        out = metrics.clustering_metrics(X, np.array([0] * 5 + [1] * 5))
        assert np.isnan(out["ari"]) and np.isnan(out["nmi"])


# ==========================================================================
# Datasets
# ==========================================================================


class TestDatasetRegistry:
    def test_available_datasets_is_sorted_and_unique(self):
        names = datasets.available_datasets()
        assert names == sorted(set(names))
        assert len(names) > 0

    def test_available_groups(self):
        assert set(datasets.available_groups()) == {"real", "synthetic", "openml"}

    def test_datasets_in_group(self):
        assert "iris" in datasets.datasets_in_group("real")
        assert "blobs" in datasets.datasets_in_group("synthetic")

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown group"):
            datasets.datasets_in_group("nope")

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            datasets.get_dataset("definitely_not_a_dataset")


@needs_sklearn
class TestDatasetLoading:
    """Only datasets that ship with scikit-learn or are generated locally —
    nothing here touches the network."""

    @pytest.mark.parametrize("name", ["iris", "wine", "digits", "breast_cancer"])
    def test_bundled_real_datasets(self, name):
        X, y = datasets.get_dataset(name)
        assert X.ndim == 2 and X.shape[0] == y.shape[0]

    @pytest.mark.parametrize(
        "name",
        [
            "blobs",
            "moons",
            "circles",
            "varied_blobs",
            "anisotropic_blobs",
            "high_dimensional_blobs",
        ],
    )
    def test_synthetic_datasets(self, name):
        X, y = datasets.get_dataset(name)
        assert X.ndim == 2 and X.shape[0] == y.shape[0]

    def test_synthetic_datasets_are_reproducible(self):
        X1, _ = datasets.get_dataset("blobs", random_state=7)
        X2, _ = datasets.get_dataset("blobs", random_state=7)
        assert np.array_equal(X1, X2)

    def test_name_is_case_insensitive(self):
        X_lower, _ = datasets.get_dataset("iris")
        X_upper, _ = datasets.get_dataset("IRIS")
        assert np.array_equal(X_lower, X_upper)

    def test_dataset_info(self):
        info = datasets.dataset_info("iris")
        assert info == {
            "name": "iris",
            "n_samples": 150,
            "n_features": 4,
            "n_classes": 3,
        }

    def test_benchmark_suite_synthetic(self):
        suite = datasets.benchmark_suite("synthetic")
        assert set(suite) == set(datasets.datasets_in_group("synthetic"))
        for X, y in suite.values():
            assert X.shape[0] == y.shape[0]


# ==========================================================================
# Benchmark backends
# ==========================================================================


class TestRuntimeBenchmark:
    def test_reports_fit_and_predict_times(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter, RuntimeBenchmark

        out = RuntimeBenchmark(n_repeats=2).evaluate(
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X
        )
        assert out["fit_time_sec"] >= 0
        assert out["fit_time_std"] >= 0
        assert "predict_time_sec" in out

    def test_callable_shorthand(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter, RuntimeBenchmark

        bench = RuntimeBenchmark(n_repeats=1)
        model = BenchmarkAdapter(FitPredictWithK(), n_clusters=2)
        assert set(bench(model, X)) == set(bench.evaluate(model, X))

    def test_failing_predict_is_tolerated(self, X):
        from soft_clustering.benchmarking import RuntimeBenchmark

        out = RuntimeBenchmark(n_repeats=1).evaluate(PredictRaises(), X)
        assert np.isfinite(out["predict_time_sec"])

    def test_model_without_predict_reports_nan(self, X):
        from soft_clustering.benchmarking import RuntimeBenchmark

        out = RuntimeBenchmark(n_repeats=1).evaluate(LabelsOnly(), X)
        assert np.isnan(out["predict_time_sec"])


@needs_psutil
class TestMemoryBenchmark:
    def test_reports_memory_and_time(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter, MemoryBenchmark

        out = MemoryBenchmark().evaluate(
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X
        )
        assert out["memory_after_mb"] > 0
        assert out["fit_time_sec"] >= 0
        assert out["memory_delta_mb"] == pytest.approx(
            out["memory_after_mb"] - out["memory_before_mb"], abs=1e-3
        )

    def test_peak_is_never_below_the_endpoint_readings(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter, MemoryBenchmark

        out = MemoryBenchmark().evaluate(
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X
        )
        assert out["peak_memory_mb"] >= out["memory_before_mb"]
        assert out["peak_memory_mb"] >= out["memory_after_mb"]

    def test_poll_interval_is_actually_used(self):
        """The sampler must run: a fit long enough to span several intervals
        has to yield samples. Before this was implemented, poll_interval was
        accepted and documented but never read."""
        from soft_clustering.benchmarking import MemoryBenchmark

        class SlowFit:
            def fit(self, X):
                time.sleep(0.25)
                return self

        out = MemoryBenchmark(poll_interval=0.01).evaluate(SlowFit(), None)
        assert out["n_samples_taken"] >= 5, out

    def test_peak_captures_a_transient_allocation(self):
        """A large array allocated and freed inside fit() is invisible to the
        before/after readings, and must be caught by the sampler."""
        from soft_clustering.benchmarking import MemoryBenchmark

        class TransientAllocation:
            def fit(self, X):
                for _ in range(12):
                    block = np.ones((1024, 1024))  # 8 MB, released each pass
                    time.sleep(0.01)
                    del block
                return self

        out = MemoryBenchmark(poll_interval=0.005).evaluate(TransientAllocation(), None)
        assert out["n_samples_taken"] > 0
        assert out["peak_memory_mb"] > out["memory_after_mb"]

    def test_rejects_a_non_positive_poll_interval(self):
        from soft_clustering.benchmarking import MemoryBenchmark

        with pytest.raises(ValueError, match="poll_interval must be positive"):
            MemoryBenchmark(poll_interval=0)

    def test_sampler_stops_after_evaluate(self, X):
        from soft_clustering.benchmarking import BenchmarkAdapter, MemoryBenchmark

        before = threading.active_count()
        MemoryBenchmark().evaluate(BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X)
        assert threading.active_count() == before

    def test_sampler_stops_when_fit_raises(self):
        from soft_clustering.benchmarking import MemoryBenchmark

        class Explodes:
            def fit(self, X):
                raise RuntimeError("boom")

        before = threading.active_count()
        with pytest.raises(RuntimeError, match="boom"):
            MemoryBenchmark().evaluate(Explodes(), None)
        assert threading.active_count() == before


@needs_psutil
class TestScalabilityBenchmark:
    def test_measures_each_requested_size(self, X):
        from soft_clustering.benchmarking import (
            BenchmarkAdapter,
            ScalabilityBenchmark,
        )

        out = ScalabilityBenchmark(sample_sizes=(10, 20)).evaluate(
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X
        )
        assert set(out) == {"runtime_10", "runtime_20", "memory_10", "memory_20"}

    def test_sizes_larger_than_the_dataset_are_skipped(self, X):
        from soft_clustering.benchmarking import (
            BenchmarkAdapter,
            ScalabilityBenchmark,
        )

        out = ScalabilityBenchmark(sample_sizes=(10_000,)).evaluate(
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X
        )
        assert out == {}


class TestClusteringQualityBenchmark:
    def test_soft_metrics_are_reported(self, X):
        from soft_clustering.benchmarking import (
            BenchmarkAdapter,
            ClusteringQualityBenchmark,
        )

        out = ClusteringQualityBenchmark().evaluate(
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X
        )
        assert "partition_coefficient" in out
        assert "partition_entropy" in out

    @needs_sklearn
    def test_hard_and_supervised_metrics_are_reported(self, X):
        from soft_clustering.benchmarking import (
            BenchmarkAdapter,
            ClusteringQualityBenchmark,
        )

        y = np.array([0] * 15 + [1] * 15)
        out = ClusteringQualityBenchmark().evaluate(
            BenchmarkAdapter(FitPredictWithK(), n_clusters=2), X, y
        )
        assert {"silhouette", "calinski_harabasz", "davies_bouldin"} <= set(out)
        assert {"ari", "nmi"} <= set(out)

    def test_unusable_model_raises_a_directed_error(self, X):
        from soft_clustering.benchmarking import ClusteringQualityBenchmark

        with pytest.raises(ValueError, match="BenchmarkAdapter"):
            ClusteringQualityBenchmark().evaluate(FitNoMembership(), X)

    @needs_sklearn
    @pytest.mark.parametrize(
        "model_cls",
        [PredictRaises, LabelsOnly, MembershipOnly],
        ids=lambda c: c.__name__,
    )
    def test_label_fallback_chain(self, model_cls, X):
        """Labels are taken from predict(), then labels_, then argmax(U)."""
        from soft_clustering.benchmarking import ClusteringQualityBenchmark

        out = ClusteringQualityBenchmark().evaluate(model_cls(), X)
        assert "silhouette" in out


# ==========================================================================
# Runner and report
# ==========================================================================


@needs_pandas
class TestClusteringBenchmark:
    def test_run_returns_one_row_per_model(self, X):
        from soft_clustering.benchmarking import (
            BenchmarkAdapter,
            ClusteringBenchmark,
            RuntimeBenchmark,
        )

        df = ClusteringBenchmark(
            models=[
                BenchmarkAdapter(FitPredictWithK(), n_clusters=2, name="A"),
                BenchmarkAdapter(FitOnly(), name="B"),
            ],
            benchmarks=[RuntimeBenchmark(n_repeats=1)],
        ).run(X)

        assert list(df["model"]) == ["A", "B"]
        assert "fit_time_sec" in df.columns

    def test_metrics_from_several_backends_are_merged(self, X):
        from soft_clustering.benchmarking import (
            BenchmarkAdapter,
            ClusteringBenchmark,
            ClusteringQualityBenchmark,
            RuntimeBenchmark,
        )

        df = ClusteringBenchmark(
            models=[BenchmarkAdapter(FitPredictWithK(), n_clusters=2)],
            benchmarks=[RuntimeBenchmark(n_repeats=1), ClusteringQualityBenchmark()],
        ).run(X)

        assert "fit_time_sec" in df.columns
        assert "partition_coefficient" in df.columns
        assert len(df) == 1

    def test_unwrapped_model_is_named_by_its_class(self, X):
        from soft_clustering.benchmarking import ClusteringBenchmark, RuntimeBenchmark

        df = ClusteringBenchmark(
            models=[FitOnly()],
            benchmarks=[RuntimeBenchmark(n_repeats=1)],
        ).run(X)
        assert df["model"].iloc[0] == "FitOnly"


@needs_pandas
class TestBenchmarkReport:
    @pytest.fixture
    def results(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {"model": "A", "ari": 0.9, "fit_time_sec": 0.2},
                {"model": "B", "ari": 0.4, "fit_time_sec": 0.1},
            ]
        )

    def test_to_csv_roundtrip(self, results, tmp_path):
        import pandas as pd

        from soft_clustering.benchmarking import BenchmarkReport

        path = tmp_path / "results.csv"
        BenchmarkReport(results).to_csv(str(path))
        assert pd.read_csv(path).shape == results.shape

    @needs_tabulate
    def test_to_markdown(self, results, tmp_path):
        from soft_clustering.benchmarking import BenchmarkReport

        path = tmp_path / "results.md"
        BenchmarkReport(results).to_markdown(str(path))
        assert "model" in path.read_text()

    def test_leaderboard_orders_by_metric(self, results):
        from soft_clustering.benchmarking import BenchmarkReport

        board = BenchmarkReport(results).leaderboard("ari")
        assert list(board["model"]) == ["A", "B"]

    def test_leaderboard_ascending(self, results):
        from soft_clustering.benchmarking import BenchmarkReport

        board = BenchmarkReport(results).leaderboard("fit_time_sec", ascending=True)
        assert list(board["model"]) == ["B", "A"]

    def test_summary(self, results):
        from soft_clustering.benchmarking import BenchmarkReport

        assert "ari" in BenchmarkReport(results).summary().columns


@needs_pandas
class TestBaseBenchmark:
    def test_validate_result_rejects_non_dict(self):
        from soft_clustering.benchmarking import BaseBenchmark

        with pytest.raises(TypeError, match="dictionary"):
            BaseBenchmark.validate_result([1, 2, 3])

    def test_validate_result_accepts_dict(self):
        from soft_clustering.benchmarking import BaseBenchmark

        assert BaseBenchmark.validate_result({"a": 1}) is None

    def test_to_dataframe(self):
        from soft_clustering.benchmarking import BaseBenchmark

        df = BaseBenchmark.to_dataframe([{"a": 1}, {"a": 2}])
        assert list(df["a"]) == [1, 2]

    def test_abstract_evaluate_refuses_to_run(self):
        from soft_clustering.benchmarking import BaseBenchmark

        class Passthrough(BaseBenchmark):
            def evaluate(self, model, X, y=None):
                return super().evaluate(model, X, y)

        with pytest.raises(NotImplementedError):
            Passthrough().evaluate(None, None)


# ==========================================================================
# End-to-end with a real estimator
# ==========================================================================


class TestAdapterOnSCPPEstimators:
    """SCPP estimators already satisfy the protocol, so the adapter delegates
    to it rather than re-deriving the fitted state by signature inspection."""

    @pytest.fixture
    def features(self):
        rng = np.random.default_rng(0)
        return np.vstack(
            [rng.normal([0, 0], 0.3, (15, 2)), rng.normal([5, 5], 0.3, (15, 2))]
        )

    def test_delegates_to_the_protocol(self, features):
        from soft_clustering import FCM
        from soft_clustering.benchmarking import BenchmarkAdapter

        model = FCM(n_clusters=2, random_state=0)
        adapter = BenchmarkAdapter(model).fit(features)

        # The adapter reports exactly what the protocol published.
        assert adapter.membership_ is model.memberships_
        assert adapter.labels_ is model.labels_
        assert adapter.centers_ is model.centers_

    def test_n_clusters_may_be_given_to_the_adapter_instead(self, features):
        from soft_clustering import FCM
        from soft_clustering.benchmarking import BenchmarkAdapter

        adapter = BenchmarkAdapter(FCM(random_state=0), n_clusters=3).fit(features)
        assert adapter.membership_.shape == (30, 3)

    def test_scpp_estimators_need_no_adapter_at_all(self, features):
        """ClusteringBenchmark accepts a bare estimator; the adapter is only
        needed to relabel a model or to wrap a foreign one."""
        from soft_clustering import FCM
        from soft_clustering.benchmarking import ClusteringQualityBenchmark

        out = ClusteringQualityBenchmark().evaluate(
            FCM(n_clusters=2, random_state=0), features
        )
        assert "partition_coefficient" in out


@needs_pandas
@needs_sklearn
class TestEndToEnd:
    def test_fcm_on_iris(self):
        from soft_clustering import FCM
        from soft_clustering.benchmarking import (
            BenchmarkAdapter,
            BenchmarkReport,
            ClusteringBenchmark,
            ClusteringQualityBenchmark,
            RuntimeBenchmark,
            get_dataset,
        )

        X, y = get_dataset("iris")
        df = ClusteringBenchmark(
            models=[
                BenchmarkAdapter(
                    FCM(random_state=0, max_iter=50), n_clusters=3, name="FCM"
                )
            ],
            benchmarks=[RuntimeBenchmark(n_repeats=1), ClusteringQualityBenchmark()],
        ).run(X, y)

        assert df["model"].iloc[0] == "FCM"
        assert df["ari"].iloc[0] > 0.5  # FCM recovers the iris structure
        assert 0.0 <= df["partition_coefficient"].iloc[0] <= 1.0
        assert BenchmarkReport(df).leaderboard("ari").shape[0] == 1
