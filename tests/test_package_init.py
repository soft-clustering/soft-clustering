# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Tests for package-level imports and __init__.py."""

import importlib.util

import pytest

import soft_clustering

HAVE_DEEP = importlib.util.find_spec("torch") is not None


def test_version():
    assert hasattr(soft_clustering, "__version__")
    assert isinstance(soft_clustering.__version__, str)


def test_all_symbols_importable():
    expected = [
        "AFCM",
        "AFCMAdaptive",
        "AFCMSimple",
        "BGMM",
        "BIGCLAM",
        "BayesianNMF",
        "CAFCM",
        "CAFHFCM",
        "CDCGS",
        "DMoN",
        "ECM",
        "ENTROPYFCM",
        "FCC",
        "FCM",
        "FeMIFuzzy",
        "GK",
        "GMM",
        "KFCCL",
        "KFCM",
        "KMART",
        "LDA",
        "MBMM",
        "MMSB",
        "NOCD",
        "PCM",
        "PFCM",
        "PLSI",
        "RDFKC",
        "RoughKMeans",
        "RPFKM",
        "SCM",
        "SCSPA",
        "SFCMEP",
        "SHBGF",
        "SISC",
        "SKFCM",
        "SMCLA",
        "SoftDBSCANGM",
        "SoftKSC",
        "WBSC",
    ]
    for name in expected:
        if name in soft_clustering.DEEP_ESTIMATORS and not HAVE_DEEP:
            continue  # covered by test_deep_estimators_explain_the_missing_extra
        assert hasattr(soft_clustering, name), f"{name} not exported"


def test_all_list_matches_exports():
    for name in soft_clustering.__all__:
        if name in soft_clustering.DEEP_ESTIMATORS and not HAVE_DEEP:
            continue
        assert hasattr(soft_clustering, name)


@pytest.mark.skipif(HAVE_DEEP, reason="only meaningful without the deep extra")
def test_deep_estimators_explain_the_missing_extra():
    for name in soft_clustering.DEEP_ESTIMATORS:
        with pytest.raises(ImportError, match=r"soft-clustering\[deep\]"):
            getattr(soft_clustering, name)


@pytest.mark.parametrize(
    "long_name, short_name",
    [
        ("FuzzyCMeans", "FCM"),
        ("GustafsonKessel", "GK"),
        ("GaussianMixtureEM", "GMM"),
        ("PossibilisticCMeans", "PCM"),
    ],
)
def test_class_names_are_importable_aliases(long_name, short_name):
    assert getattr(soft_clustering, long_name) is getattr(soft_clustering, short_name)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        soft_clustering.NotAnEstimator  # noqa: B018


def test_not_fitted_error_is_both_runtime_and_sklearn_error():
    from sklearn.exceptions import NotFittedError as SklearnNotFittedError

    assert issubclass(soft_clustering.NotFittedError, RuntimeError)
    assert issubclass(soft_clustering.NotFittedError, SklearnNotFittedError)
    with pytest.raises(soft_clustering.NotFittedError, match="not fitted"):
        soft_clustering.FCM().predict()


def test_algorithm_info_records():
    info = soft_clustering.list_algorithms(family="evidential")[0]
    assert isinstance(info, soft_clustering.AlgorithmInfo)
    record = info.as_dict()
    assert record["name"] == "ECM" and record["status"] == "validated"
    with pytest.raises(AttributeError):  # records are immutable
        info.status = "unverified"
