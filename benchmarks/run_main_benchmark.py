#!/usr/bin/env python3
"""The cross-family comparison SCPP exists to make possible.

Every estimator that consumes a feature matrix is run over the dataset suite
that ships inside the package, under one protocol, and scored with the same
twelve metrics. The output is the evidence for the library's central claim:
that heterogeneous soft clustering methods can be compared on equal terms
without per-method integration code.

Three things this script is careful about, because each is a way such a table
can mislead.

**Fuzziness indices are not quality scores.** The partition coefficient,
modified partition coefficient and partition entropy measure how *crisp* a
partition is. A fuzzy method that has collapsed to a hard partition scores the
best attainable value on all three. They are reported because they describe
the soft output, and they are never used to rank estimators. Estimators that
do not impose the partition constraint report ``nan`` for them rather than a
number on a different scale.

**Failures are results.** An estimator that raises, times out or returns a
degenerate partition is recorded with its status, not dropped. Silently
omitting the runs that failed is what turns a comparison into an
advertisement.

**Only comparable things are compared.** Graph, document, ensemble, image and
semi-supervised estimators do not take a feature matrix and are not run here;
``--list-modalities`` prints the full partition of the library by input type.
Putting them in the same table as the prototype methods would compare a
document clusterer on strings against a fuzzy clusterer on a matrix.

Usage
-----
    python benchmarks/run_main_benchmark.py
    python benchmarks/run_main_benchmark.py --groups real synthetic openml
    python benchmarks/run_main_benchmark.py --list-modalities

Requires::

    pip install "soft-clustering[bench]"
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import platform
import queue as queue_module
import signal
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "results"

import soft_clustering as sc  # noqa: E402
from soft_clustering.benchmarking import benchmark_suite  # noqa: E402
from soft_clustering.benchmarking.metrics import (  # noqa: E402
    clustering_metrics,
    soft_clustering_metrics,
)

# ---------------------------------------------------------------------------
# Which estimators take a plain feature matrix
# ---------------------------------------------------------------------------

#: name -> constructor kwargs, given the cluster count K. Every estimator here
#: is fitted as ``model.fit(X)`` on an ``(n, d)`` matrix.
FEATURE_ESTIMATORS: dict[str, callable] = {
    "AFCM": lambda k: {"n_clusters": k},
    "AFCMSimple": lambda k: {"n_clusters": k},
    "CAFCM": lambda k: {"n_clusters": k},
    "CAFHFCM": lambda k: {"n_clusters": k},
    "ECM": lambda k: {"n_clusters": k},
    "ENTROPYFCM": lambda k: {"n_clusters": k},
    "EVCLUS": lambda k: {"n_clusters": k, "random_state": 0},
    "FCC": lambda k: {"n_clusters": k},
    "FCM": lambda k: {"n_clusters": k, "random_state": 0},
    "GK": lambda k: {"n_clusters": k, "random_state": 0},
    "GMM": lambda k: {"n_clusters": k, "random_state": 0},
    "GathGeva": lambda k: {"n_clusters": k, "random_state": 0},
    "KFCCL": lambda k: {"n_clusters": k},
    "KFCM": lambda k: {"n_clusters": k},
    "PCM": lambda k: {"n_clusters": k, "random_state": 0},
    "PFCM": lambda k: {"n_clusters": k, "random_state": 0},
    "RPFKM": lambda k: {"n_clusters": k, "d": 2, "random_state": 0},
    "RoughKMeans": lambda k: {"n_clusters": k, "random_state": 0},
    "SCM": lambda k: {},  # discovers K itself
    "SoftDBSCANGM": lambda k: {},  # discovers K itself
}

#: Input modality of every exported estimator, so the exclusions above are
#: stated rather than implied.
MODALITIES: dict[str, str] = {
    **{name: "feature matrix" for name in FEATURE_ESTIMATORS},
    "MBMM": "feature matrix in (0, 1)",
    "SFCMEP": "feature matrix + partial labels",
    "BGMM": "two aligned views",
    "SoftKSC": "labelled + unlabelled parts",
    "FeMIFuzzy": "list of client matrices",
    "AFCMAdaptive": "grayscale image",
    "SKFCM": "pixels + image shape",
    "BIGCLAM": "adjacency matrix",
    "BayesianNMF": "adjacency matrix",
    "MMSB": "adjacency matrix",
    "CDCGS": "adjacency matrix",
    "DMoN": "node features + graph",
    "NOCD": "sparse graph + features",
    "RDFKC": "feature matrix (torch autoencoder)",
    "KMART": "documents",
    "LDA": "documents",
    "PLSI": "documents",
    "SISC": "documents",
    "WBSC": "documents",
    "SCSPA": "ensemble of partitions",
    "SHBGF": "ensemble of partitions",
    "SMCLA": "ensemble of partitions",
}

METRIC_COLUMNS = [
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "ari",
    "nmi",
    "partition_coefficient",
    "modified_partition_coefficient",
    "partition_entropy",
    "fuzzy_hypervolume",
    "xie_beni",
    "fuzzy_compactness",
    "fuzzy_separation",
]


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def run_one(name: str, X: np.ndarray, y: np.ndarray, k: int, timeout_s: float) -> dict:
    """Fit one estimator in a subprocess and score it.

    The subprocess is not for measurement hygiene but for survivability: some
    estimator/dataset pairs are genuinely infeasible rather than merely slow.
    Gath--Geva on ``olivetti_faces`` would allocate a 4096x4096 covariance for
    each of 40 clusters, about 5 GB. Isolating the fit means such a pair is
    recorded as a timeout, the way the optimization study records them, rather
    than taking the whole run down.
    """
    blank = {
        "algorithm": name,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "k_requested": int(k),
        "k_fitted": None,
        "status": "timeout",
        "error": f"did not finish within {timeout_s:.0f} s",
        "fit_time_ms": float("nan"),
        "peak_traced_mb": float("nan"),
        **dict.fromkeys(METRIC_COLUMNS, float("nan")),
    }

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=_fit_and_score, args=(queue, name, X, y, k))
    worker.start()

    # Read the result *before* joining. A child that has put an item on the
    # queue may not have flushed the feeder thread by the time it exits, so
    # join-then-get_nowait races and reports a successful run as a crash.
    try:
        result = queue.get(timeout=timeout_s)
    except queue_module.Empty:
        result = None

    if worker.is_alive():
        worker.terminate()
    worker.join()

    if result is None:
        if worker.exitcode not in (0, -signal.SIGTERM):
            blank["status"] = "crashed"
            blank["error"] = f"worker exited with code {worker.exitcode}"
        return blank
    return result


def _fit_and_score(queue, name: str, X: np.ndarray, y: np.ndarray, k: int) -> None:
    """Subprocess body: fit, score, put one row on the queue."""
    row = {
        "algorithm": name,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "k_requested": int(k),
        "k_fitted": None,
        "status": "ok",
        "error": "",
        "fit_time_ms": float("nan"),
        "peak_traced_mb": float("nan"),
        **dict.fromkeys(METRIC_COLUMNS, float("nan")),
    }

    try:
        estimator = getattr(sc, name)(**FEATURE_ESTIMATORS[name](k))
    except Exception as exc:  # pragma: no cover - construction should not fail
        row["status"], row["error"] = "construct_error", f"{type(exc).__name__}: {exc}"
        queue.put(row)
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tracemalloc.start()
            start = time.perf_counter()
            estimator.fit(X)
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    except Exception as exc:
        tracemalloc.stop()
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        queue.put(row)
        return

    row["fit_time_ms"] = elapsed * 1000.0
    row["peak_traced_mb"] = peak / 1024**2

    U = estimator.memberships_
    if U is None or U.shape[1] == 0:
        row["status"] = "empty_partition"
        queue.put(row)
        return

    labels = estimator.labels_
    row["k_fitted"] = int(U.shape[1])
    if len(np.unique(labels)) < 2:
        row["status"] = "degenerate"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row.update(clustering_metrics(X, labels, y_true=y))
        row.update(
            soft_clustering_metrics(
                X,
                U,
                centers=estimator.centers_,
                m=2.0,
                partition_constrained=estimator._partition_constrained,
            )
        )
    queue.put(row)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarise(rows: list[dict]) -> list[dict]:
    """Per-algorithm means over the datasets where the fit succeeded."""
    summary = []
    for name in sorted({r["algorithm"] for r in rows}):
        mine = [r for r in rows if r["algorithm"] == name]
        ok = [r for r in mine if r["status"] in ("ok", "slow")]
        entry = {
            "algorithm": name,
            "datasets": len(mine),
            "completed": len(ok),
            "degenerate": sum(r["status"] == "degenerate" for r in mine),
            "errors": sum(r["status"] in ("error", "construct_error") for r in mine),
        }
        for column in ("ari", "nmi", "silhouette", "fit_time_ms", "peak_traced_mb"):
            values = [r[column] for r in ok if np.isfinite(r[column])]
            entry[f"mean_{column}"] = float(np.mean(values)) if values else float("nan")
        summary.append(entry)
    return summary


def write_outputs(rows: list[dict], summary: list[dict], groups: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "main_benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (OUT / "main_benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    (OUT / "main_benchmark_env.json").write_text(
        json.dumps(
            {
                "groups": groups,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "sklearn": __import__("sklearn").__version__,
                "soft_clustering": sc.__version__,
            },
            indent=2,
        )
    )

    def cell(value, spec=".3f"):
        return "---" if not np.isfinite(value) else format(value, spec)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-family comparison over the dataset suite that ships "
        r"with SCPP. Every estimator that consumes a feature matrix is fitted "
        r"on every dataset under one protocol. \emph{Completed} counts the "
        r"datasets on which the fit returned a non-degenerate partition; "
        r"failures are reported, not dropped. ARI, NMI and silhouette are "
        r"means over the completed runs. The fuzziness indices are reported "
        r"per run in the released CSV and deliberately omitted here: they "
        r"measure crispness, not quality, and ranking estimators by them "
        r"rewards a fuzzy method for degenerating to a hard partition.}",
        r"\label{tab:main-benchmark}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Algorithm & Datasets & Compl. & Degen. & Err. & ARI $\uparrow$ & "
        r"NMI $\uparrow$ & $\bar{t}$ (ms) \\",
        r"\midrule",
    ]
    for entry in sorted(summary, key=lambda e: -(e["mean_ari"] if np.isfinite(e["mean_ari"]) else -9)):
        lines.append(
            f"{entry['algorithm']} & {entry['datasets']} & {entry['completed']} & "
            f"{entry['degenerate']} & {entry['errors']} & "
            f"{cell(entry['mean_ari'])} & {cell(entry['mean_nmi'])} & "
            f"{cell(entry['mean_fit_time_ms'], '.1f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "main_benchmark.tex").write_text("\n".join(lines))

    for path in (
        "main_benchmark.csv",
        "main_benchmark_summary.csv",
        "main_benchmark.tex",
        "main_benchmark_env.json",
    ):
        print(f"wrote {OUT / path}")


def print_modalities() -> None:
    by_modality: dict[str, list[str]] = {}
    for name in sorted(sc._ESTIMATORS):
        by_modality.setdefault(MODALITIES.get(name, "UNCLASSIFIED"), []).append(name)
    total = 0
    print(f"{len(sc._ESTIMATORS)} exported estimators by input modality:\n")
    for modality, names in sorted(by_modality.items()):
        total += len(names)
        print(f"  {modality:36s} {len(names):2d}  {', '.join(names)}")
    assert total == len(sc._ESTIMATORS), "modality table is out of step with the registry"
    print(
        f"\n{len(FEATURE_ESTIMATORS)} of them take a plain feature matrix and are "
        "compared by this script."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["real", "synthetic"],
        choices=["real", "synthetic", "openml"],
        help="dataset groups to use; openml downloads over the network",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="subsample larger datasets, so one estimator "
                             "cannot dominate the wall time")
    parser.add_argument("--list-modalities", action="store_true")
    args = parser.parse_args()

    if args.list_modalities:
        print_modalities()
        return 0

    datasets: dict[str, tuple] = {}
    for group in args.groups:
        datasets.update(benchmark_suite(group))

    rng = np.random.default_rng(0)
    rows: list[dict] = []

    for dataset_name, (X, y) in sorted(datasets.items()):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.shape[0] > args.max_samples:
            keep = rng.choice(X.shape[0], args.max_samples, replace=False)
            X, y = X[keep], y[keep]
        k = int(len(np.unique(y)))

        print(f"\n{dataset_name}  n={X.shape[0]} d={X.shape[1]} K={k}")
        for name in sorted(FEATURE_ESTIMATORS):
            row = run_one(name, X, y, k, args.timeout)
            row["dataset"] = dataset_name
            rows.append(row)
            ari = row["ari"]
            flag = "" if row["status"] == "ok" else f"  [{row['status']}]"
            print(
                f"   {name:14s} ARI={ari:6.3f}  t={row['fit_time_ms']:8.1f} ms{flag}"
                if np.isfinite(ari)
                else f"   {name:14s} ARI=   ---  t={row['fit_time_ms']:8.1f} ms{flag}"
            )

    summary = summarise(rows)
    print("\n" + "=" * 72)
    print(f"{'algorithm':16s} {'compl.':>7s} {'degen.':>7s} {'err':>4s} "
          f"{'mean ARI':>9s} {'mean t (ms)':>12s}")
    print("-" * 72)
    for entry in sorted(summary, key=lambda e: -(e["mean_ari"] if np.isfinite(e["mean_ari"]) else -9)):
        print(
            f"{entry['algorithm']:16s} {entry['completed']:7d} "
            f"{entry['degenerate']:7d} {entry['errors']:4d} "
            f"{entry['mean_ari']:9.3f} {entry['mean_fit_time_ms']:12.1f}"
        )

    write_outputs(rows, summary, args.groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
