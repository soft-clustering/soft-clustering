"""Tests for package-level imports and __init__.py."""
import soft_clustering


def test_version():
    assert hasattr(soft_clustering, "__version__")
    assert isinstance(soft_clustering.__version__, str)


def test_all_symbols_importable():
    expected = [
        "AFCM", "AFCMAdaptive", "AFCMSimple", "BGMM", "BIGCLAM",
        "BayesianNMF", "CAFCM", "CAFHFCM", "CDCGS", "DMoN", "ECM",
        "ENTROPYFCM", "FCC", "FCM", "FeMIFuzzy", "GK", "GMM", "KFCCL",
        "KFCM", "KMART", "LDA", "MBMM", "MMSB", "NOCD", "PCM", "PFCM",
        "PLSI", "RDFKC", "RoughKMeans", "RPFKM", "SCM", "SCSPA", "SFCMEP",
        "SHBGF", "SISC", "SKFCM", "SMCLA", "SoftDBSCANGM", "SoftKSC", "WBSC",
    ]
    for name in expected:
        assert hasattr(soft_clustering, name), f"{name} not exported"


def test_all_list_matches_exports():
    for name in soft_clustering.__all__:
        assert hasattr(soft_clustering, name)
