#!/usr/bin/env python3
"""Compare SCPP against independent implementations of the same algorithms.

The optimization study in ``optimization/`` measures SCPP against SCPP: it
reports how much faster the current implementations are than the ones this
project shipped before. That answers "did the rewrite help", not "is this
library worth using instead of what already exists". This script answers the
second question for the algorithms where an independent Python implementation
exists.

For each pair it reports

* whether the two implementations reach the same solution, measured as the
  maximum absolute membership difference after matching cluster labels, and
  the adjusted Rand index between the hard partitions; and
* the runtime of each, under one protocol --- one warm-up fit, then the
  minimum of ``--repeats`` timed fits, which is the protocol the optimization
  appendix uses.

Agreement is the point. A speed difference against a mature library is worth
knowing but is not the contribution; being demonstrably the *same algorithm*
is what lets a reader trust the other 28 estimators that have no reference to
check against.

Usage
-----
    python benchmarks/run_external_baselines.py
    python benchmarks/run_external_baselines.py --sizes 200 1000 --repeats 5

Requires the ``baselines`` extra::

    pip install "soft-clustering[bench,baselines]"
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def blobs(n: int, d: int = 8, k: int = 3, spread: float = 0.45, seed: int = 0):
    rng = np.random.default_rng(seed)
    per = n // k
    parts = [rng.normal(3.0 * i, spread, (per, d)) for i in range(k - 1)]
    parts.append(rng.normal(3.0 * (k - 1), spread, (n - per * (k - 1), d)))
    return np.vstack(parts), np.repeat(
        np.arange(k), [per] * (k - 1) + [n - per * (k - 1)]
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def timed(fn, repeats: int) -> float:
    """Minimum wall time over ``repeats`` runs after one warm-up, in ms."""
    fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Permute ``candidate``'s columns to best match ``reference``."""
    remaining = list(range(candidate.shape[1]))
    order = []
    for column in reference.T:
        best = max(remaining, key=lambda j: float(np.dot(column, candidate[:, j])))
        order.append(best)
        remaining.remove(best)
    return candidate[:, order]


@dataclass
class Row:
    algorithm: str
    reference: str
    n: int
    d: int
    k: int
    max_membership_diff: float
    label_ari: float
    scpp_ms: float
    reference_ms: float
    speed_ratio: float  # reference / scpp; > 1 means SCPP is faster


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


def compare_fcm(n, d, k, repeats):
    import skfuzzy

    from soft_clustering import FCM

    X, _ = blobs(n, d, k)
    kwargs = dict(n_clusters=k, m=2.0, max_iter=1000, tol=1e-10, random_state=0)
    reference_kwargs = dict(c=k, m=2.0, error=1e-10, maxiter=1000, seed=0)

    ours = FCM(**kwargs).fit(X)
    centres, u, *_ = skfuzzy.cluster.cmeans(X.T, **reference_kwargs)

    return _row(
        "FCM",
        "scikit-fuzzy cmeans",
        X,
        k,
        ours.memberships_,
        u.T,
        lambda: FCM(**kwargs).fit(X),
        lambda: skfuzzy.cluster.cmeans(X.T, **reference_kwargs),
        repeats,
    )


def compare_gmm(n, d, k, repeats):
    from sklearn.mixture import GaussianMixture

    from soft_clustering import GMM

    X, _ = blobs(n, d, k)
    kwargs = dict(n_clusters=k, max_iter=500, tol=1e-10, random_state=0)
    reference_kwargs = dict(
        n_components=k, covariance_type="full", max_iter=500, tol=1e-10, random_state=0
    )

    ours = GMM(**kwargs).fit(X)
    reference = GaussianMixture(**reference_kwargs).fit(X)

    return _row(
        "GMM",
        "scikit-learn GaussianMixture",
        X,
        k,
        ours.memberships_,
        reference.predict_proba(X),
        lambda: GMM(**kwargs).fit(X),
        lambda: GaussianMixture(**reference_kwargs).fit(X),
        repeats,
    )


def _row(name, reference_name, X, k, ours_U, theirs_U, ours_fn, theirs_fn, repeats):
    from sklearn.metrics import adjusted_rand_score

    theirs_U = align(ours_U, theirs_U)
    return Row(
        algorithm=name,
        reference=reference_name,
        n=X.shape[0],
        d=X.shape[1],
        k=k,
        max_membership_diff=float(np.abs(ours_U - theirs_U).max()),
        label_ari=float(
            adjusted_rand_score(np.argmax(ours_U, 1), np.argmax(theirs_U, 1))
        ),
        scpp_ms=timed(ours_fn, repeats),
        reference_ms=timed(theirs_fn, repeats),
        speed_ratio=0.0,
    )


PAIRS = {"FCM": compare_fcm, "GMM": compare_gmm}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_outputs(rows: list[Row]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "external_baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": __import__("numpy").__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": __import__("sklearn").__version__,
    }
    try:
        environment["skfuzzy"] = __import__("skfuzzy").__version__
    except Exception:  # pragma: no cover
        pass
    (OUT / "external_baselines_env.json").write_text(json.dumps(environment, indent=2))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SCPP against independent implementations of the same "
        r"algorithms. Agreement is the maximum absolute difference between "
        r"membership matrices after matching cluster labels, with the adjusted "
        r"Rand index between the hard partitions. Runtimes are the minimum of "
        r"three timed fits after one warm-up.}",
        r"\label{tab:external-baselines}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Algorithm & Reference & $n$ & Max.\ diff. & ARI & SCPP (ms) & "
        r"Reference (ms) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.algorithm} & {row.reference} & {row.n:,} & "
            f"{row.max_membership_diff:.1e} & {row.label_ari:.4f} & "
            f"{row.scpp_ms:.1f} & {row.reference_ms:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "external_baselines.tex").write_text("\n".join(lines))

    print(f"\nwrote {OUT / 'external_baselines.csv'}")
    print(f"wrote {OUT / 'external_baselines.tex'}")
    print(f"wrote {OUT / 'external_baselines_env.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 1000, 5000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dims", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args()

    rows: list[Row] = []
    header = (
        f"{'algorithm':10s} {'n':>6s} {'max diff':>10s} {'ARI':>7s} "
        f"{'SCPP ms':>10s} {'ref ms':>10s}"
    )
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        for name, compare in PAIRS.items():
            try:
                row = compare(n, args.dims, args.clusters, args.repeats)
            except ImportError as exc:
                print(f"  skipping {name}: {exc}")
                continue
            row.speed_ratio = row.reference_ms / row.scpp_ms
            rows.append(row)
            print(
                f"{row.algorithm:10s} {row.n:6d} {row.max_membership_diff:10.2e} "
                f"{row.label_ari:7.4f} {row.scpp_ms:10.1f} {row.reference_ms:10.1f}"
            )

    if not rows:
        print("no comparisons ran; install the `baselines` extra")
        return 1

    write_outputs(rows)
    worst = max(r.max_membership_diff for r in rows)
    print(f"\nworst membership disagreement across all pairs: {worst:.2e}")
    print(f"all label partitions identical: {all(r.label_ari == 1.0 for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
