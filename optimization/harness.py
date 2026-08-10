"""Shared experiment harness for the SCPP optimization study.

One registry of runnable, *scalable* estimator configurations, used by the
profiler, the benchmark runner and the correctness checker alike. Keeping a
single registry is what makes the three consistent: an algorithm is profiled,
optimized and verified on exactly the same inputs.

The configurations mirror ``tests/test_protocol.py``'s ``CASES`` — same input
modality per estimator — but are parameterised by size so that scalability can
be swept. Estimators that ``CASES`` excludes for reasons of input shape are
excluded here too, with the same reasons.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Input builders, one per modality SCPP accepts.
# --------------------------------------------------------------------------

_WORDS = (
    "fuzzy clustering membership degree centroid distance metric kernel "
    "graph community detection nodes network overlapping topic document "
    "word probabilistic mixture gaussian expectation maximization soft "
    "assignment spectral possibilistic evidential rough consensus ensemble"
).split()


def features(n: int, d: int, k: int, seed: int = 0) -> np.ndarray:
    """k well-separated isotropic blobs in d dimensions."""
    rng = np.random.default_rng(seed)
    per = n // k
    parts = [rng.normal(loc=3.0 * i, scale=0.35, size=(per, d)) for i in range(k - 1)]
    parts.append(rng.normal(loc=3.0 * (k - 1), scale=0.35, size=(n - per * (k - 1), d)))
    return np.vstack(parts)


def unit_features(n: int, d: int, k: int, seed: int = 0) -> np.ndarray:
    """Samples strictly inside (0, 1) — required by Beta-mixture methods."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    return np.clip(X, 1e-3, 1 - 1e-3)


def graph(n: int, d: int, k: int, seed: int = 3, density: float = 0.3) -> np.ndarray:
    """Symmetric binary adjacency matrix with no self-loops."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < density).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def documents(n: int, d: int, k: int, seed: int = 5) -> list[str]:
    """n short documents drawn from a small vocabulary."""
    rng = np.random.default_rng(seed)
    return [
        " ".join(rng.choice(_WORDS, size=max(4, d), replace=True)) for _ in range(n)
    ]


def ensemble(n: int, d: int, k: int, seed: int = 7, n_partitions: int = 3):
    """A list of soft membership matrices, as consensus methods consume."""
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(n_partitions):
        M = np.abs(rng.normal(size=(n, k)))
        mats.append(M / M.sum(axis=1, keepdims=True))
    return mats


def partial_labels(n: int, d: int, k: int, n_labelled: int = 2) -> np.ndarray:
    """Semi-supervised target: a few labelled samples per class, rest None."""
    labels = np.full(n, None, dtype=object)
    per = n // k
    for cluster in range(k):
        start = cluster * per
        labels[start : start + n_labelled] = cluster
    return labels


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One runnable estimator configuration."""

    name: str
    family: str
    modality: str
    #: kwargs passed to the constructor; ``{k}`` placeholders filled with K
    kwargs: Callable[[int], dict[str, Any]]
    #: builds the positional fit arguments for a given (n, d, k)
    build: Callable[[int, int, int], tuple]
    #: size axes that are meaningful to sweep for this estimator
    axes: tuple[str, ...] = ("n",)
    #: default problem size (n, d, k)
    default: tuple[int, int, int] = (600, 8, 3)
    notes: str = ""
    requires_torch: bool = False


def _f(n, d, k):
    return (features(n, d, k),)


CASES: dict[str, Case] = {}


def _add(case: Case) -> None:
    CASES[case.name] = case


# --- prototype / feature-matrix methods -----------------------------------
for _name, _fam in [
    ("FCM", "fuzzy"),
    ("PCM", "possibilistic"),
    ("GK", "fuzzy"),
    ("GMM", "mixture"),
    ("PFCM", "possibilistic"),
]:
    _add(
        Case(
            _name,
            _fam,
            "features",
            lambda k: {"n_clusters": k, "random_state": 0},
            _f,
            axes=("n", "d", "k"),
        )
    )

for _name, _fam in [
    ("CAFCM", "fuzzy"),
    ("CAFHFCM", "fuzzy"),
    ("ENTROPYFCM", "fuzzy"),
    ("AFCM", "fuzzy"),
    ("AFCMSimple", "fuzzy"),
    ("FCC", "fuzzy"),
    ("KFCM", "kernel"),
    ("KFCCL", "kernel"),
    ("ECM", "evidential"),
    ("RoughKMeans", "rough"),
]:
    _add(
        Case(
            _name,
            _fam,
            "features",
            lambda k: {"n_clusters": k},
            _f,
            axes=("n", "d", "k"),
        )
    )

_add(
    Case(
        "SCM",
        "subtractive",
        "features",
        lambda k: {},
        _f,
        axes=("n", "d"),
        notes="determines K itself",
    )
)
_add(
    Case(
        "SoftDBSCANGM",
        "density",
        "features",
        lambda k: {},
        _f,
        axes=("n", "d"),
        notes="determines K itself",
    )
)
_add(
    Case(
        "RPFKM",
        "fuzzy",
        "features",
        lambda k: {"n_clusters": k, "d": 2, "random_state": 0},
        _f,
        axes=("n", "d", "k"),
    )
)
_add(
    Case(
        "MBMM",
        "mixture",
        "features",
        lambda k: {"n_clusters": k},
        lambda n, d, k: (unit_features(n, d, k),),
        axes=("n", "d", "k"),
    )
)
_add(
    Case(
        "SFCMEP",
        "semi-supervised",
        "features",
        lambda k: {"n_clusters": k},
        lambda n, d, k: (features(n, d, k), partial_labels(n, d, k)),
        axes=("n", "d", "k"),
    )
)

# --- graph methods ---------------------------------------------------------
_add(
    Case(
        "BIGCLAM",
        "graph",
        "graph",
        lambda k: {"n_clusters": k},
        lambda n, d, k: (graph(n, d, k),),
        axes=("n",),
        default=(300, 0, 3),
    )
)
_add(
    Case(
        "BayesianNMF",
        "graph",
        "graph",
        lambda k: {"n_clusters": k},
        lambda n, d, k: (graph(n, d, k),),
        axes=("n",),
        default=(300, 0, 3),
    )
)

# --- document methods ------------------------------------------------------
_add(
    Case(
        "WBSC",
        "document",
        "documents",
        lambda k: {},
        lambda n, d, k: (documents(n, d, k),),
        axes=("n",),
        default=(200, 12, 3),
        notes="determines K itself",
    )
)
_add(
    Case(
        "SISC",
        "document",
        "documents",
        lambda k: {"n_clusters": k},
        lambda n, d, k: (documents(n, d, k),),
        axes=("n",),
        default=(200, 12, 3),
    )
)
_add(
    Case(
        "KMART",
        "document",
        "documents",
        lambda k: {},
        lambda n, d, k: (documents(n, d, k),),
        axes=("n",),
        default=(200, 12, 3),
        notes="determines K itself",
    )
)
_add(
    Case(
        "PLSI",
        "document",
        "documents",
        lambda k: {"n_clusters": k, "max_iter": 20, "random_state": 0},
        lambda n, d, k: (documents(n, d, k),),
        axes=("n",),
        default=(200, 12, 3),
    )
)

# --- ensemble methods ------------------------------------------------------
for _name in ("SCSPA", "SHBGF", "SMCLA"):
    _add(
        Case(
            _name,
            "ensemble",
            "ensemble",
            lambda k: {"n_clusters": k},
            lambda n, d, k: (ensemble(n, d, k),),
            axes=("n", "k"),
        )
    )


# --------------------------------------------------------------------------
# Execution helpers
# --------------------------------------------------------------------------


def build_inputs(case: Case, n: int, d: int, k: int) -> tuple:
    return case.build(n, d, k)


def make_estimator(case: Case, k: int, module=None):
    """Instantiate ``case`` from ``module`` (defaults to soft_clustering)."""
    if module is None:
        import soft_clustering as module  # noqa: PLC0415
    return getattr(module, case.name)(**case.kwargs(k))


def run_once(case: Case, n: int, d: int, k: int, module=None):
    """Fit the estimator once and return (estimator, wall time in seconds)."""
    import time

    est = make_estimator(case, k, module)
    args = build_inputs(case, n, d, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        est.fit(*args)
        elapsed = time.perf_counter() - start
    return est, elapsed


def available_cases(include_torch: bool = False) -> dict[str, Case]:
    """Cases whose dependencies are importable in this environment."""
    out = {}
    for name, case in CASES.items():
        if case.requires_torch and not include_torch:
            continue
        out[name] = case
    return out
