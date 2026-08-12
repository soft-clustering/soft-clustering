"""Soft Clustering - A library of soft and fuzzy clustering algorithms."""

__version__ = "0.1.0"

from ._base import BaseSoftClusterer

# Estimator name -> module that defines it. Imports are deferred (PEP 562) so
# that `import soft_clustering` costs nothing for the models a user does not
# touch, and so that the five PyTorch-backed estimators do not make torch a
# hard dependency of the whole package.
_ESTIMATORS = {
    "AFCM": "._afcm",
    "AFCMAdaptive": "._afcmadaptive",
    "AFCMSimple": "._afcmSimple",
    "BGMM": "._bgmm",
    "BIGCLAM": "._bigclam",
    "BayesianNMF": "._bnmf",
    "CAFCM": "._cafcm",
    "CAFHFCM": "._cafhfcm",
    "CDCGS": "._cdcgs",
    "DMoN": "._dmon",
    "ECM": "._ecm",
    "ENTROPYFCM": "._entropyfcm",
    "EVCLUS": "._evclus",
    "FCC": "._fcc",
    "FCM": "._fcm",
    "FeMIFuzzy": "._femifuzzy",
    "GK": "._gk",
    "GMM": "._gmm",
    "GathGeva": "._gathgeva",
    "KFCCL": "._kfccl",
    "KFCM": "._kfcm",
    "KMART": "._kmart",
    "LDA": "._lda",
    "MBMM": "._mbmm",
    "MMSB": "._mmsb",
    "NOCD": "._nocd",
    "PCM": "._pcm",
    "PFCM": "._pfcm",
    "PLSI": "._plsi",
    "RDFKC": "._rdfkc",
    "RoughKMeans": "._rough_k_means",
    "RPFKM": "._rpfkm",
    "SCM": "._scm",
    "SCSPA": "._scspa",
    "SFCMEP": "._sfcmep",
    "SHBGF": "._shbgf",
    "SISC": "._sisc",
    "SKFCM": "._skfcm",
    "SMCLA": "._smcla",
    "SoftDBSCANGM": "._softdbscangm",
    "SoftKSC": "._softksc",
    "WBSC": "._wbsc",
}

# Names that differ between the module and the public API.
_ALIASES = {
    "FCM": "FuzzyCMeans",
    "GK": "GustafsonKessel",
    "GMM": "GaussianMixtureEM",
    "PCM": "PossibilisticCMeans",
}

#: Estimators requiring the optional ``deep`` extra (torch, torch_geometric).
#:
#: MMSB is deliberately absent: its variational EM is pure NumPy/SciPy, so the
#: model is a Bayesian graph model rather than a deep one and is available in
#: the base installation.
DEEP_ESTIMATORS = frozenset({"CDCGS", "DMoN", "NOCD", "RDFKC"})


def __getattr__(name):
    """Import an estimator on first access (PEP 562)."""
    module_path = _ESTIMATORS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_path, __name__)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        if name in DEEP_ESTIMATORS:
            raise ImportError(
                f"{name} requires the optional deep-learning dependencies. "
                f'Install them with: pip install "soft-clustering[deep]"'
            ) from exc
        raise

    obj = getattr(module, _ALIASES.get(name, name))
    globals()[name] = obj  # cache, so __getattr__ runs once per estimator
    return obj


def __dir__():
    return sorted(list(globals()) + list(_ESTIMATORS))


__all__ = sorted(_ESTIMATORS) + ["BaseSoftClusterer", "DEEP_ESTIMATORS"]
