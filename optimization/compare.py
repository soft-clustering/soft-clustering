#!/usr/bin/env python3
"""Compare an optimized implementation against its preserved reference.

Fits both builds on identical inputs across several sizes and seeds, then
quantifies the difference in every output that carries meaning: the membership
matrix, the hard labels, the cluster centres and the discovered cluster count.

Speed is not measured here. The only question this answers is whether the
optimized implementation computes the same thing.

Usage
-----
    python optimization/compare.py SoftDBSCANGM --sizes 60,120 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "original"))

from harness import CASES, build_inputs, make_estimator  # noqa: E402


def _fit(case, module, n, d, k, seed):
    """Fit one build; inputs are rebuilt per call so neither sees mutated data.

    Several estimators draw their initialisation from NumPy's *global* RNG
    rather than a ``random_state`` argument (MBMM is one). Seeding the legacy
    global generator immediately before each fit is what makes the two builds
    start from identical parameters, which a deterministic comparison requires.
    """
    est = make_estimator(case, k, module)
    args = build_inputs(case, n, d, k)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        np.random.seed(seed)
        est.fit(*args)
    return est


def _errors(a: np.ndarray | None, b: np.ndarray | None) -> dict:
    if a is None or b is None:
        return {"status": "absent" if a is b else "one-sided"}
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return {"status": "shape-mismatch", "a": list(a.shape), "b": list(b.shape)}
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, diff / scale, 0.0)
    return {
        "status": "compared",
        "shape": list(a.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(np.nanmax(rel)),
        "mean_rel": float(np.nanmean(rel)),
    }


def compare_one(case_name: str, n: int, d: int, k: int, seed: int) -> dict:
    import scpp_original

    import soft_clustering

    case = CASES[case_name]

    # Both builders are seeded identically inside harness, so the inputs match.
    ref = _fit(case, scpp_original, n, d, k, seed)
    opt = _fit(case, soft_clustering, n, d, k, seed)

    result = {
        "case": case_name,
        "n": n,
        "d": d,
        "k": k,
        "seed": seed,
        "memberships": _errors(
            getattr(ref, "memberships_", None), getattr(opt, "memberships_", None)
        ),
        "centers": _errors(
            getattr(ref, "centers_", None), getattr(opt, "centers_", None)
        ),
        "n_clusters_ref": getattr(ref, "n_clusters", None),
        "n_clusters_opt": getattr(opt, "n_clusters", None),
    }

    lr = getattr(ref, "labels_", None)
    lo = getattr(opt, "labels_", None)
    if lr is not None and lo is not None and np.shape(lr) == np.shape(lo):
        agree = float(np.mean(np.asarray(lr) == np.asarray(lo)))
        result["label_agreement"] = agree
        # A permuted-but-equivalent partition is still the same clustering.
        try:
            from sklearn.metrics import adjusted_rand_score

            result["label_ari"] = float(adjusted_rand_score(lr, lo))
        except ImportError:
            pass

    # Invariants that must hold regardless of the reference.
    U = getattr(opt, "memberships_", None)
    if U is not None:
        result["invariants"] = {
            "non_negative": bool((U >= 0).all()),
            "finite": bool(np.isfinite(U).all()),
            "rows_sum_to_one": bool(np.allclose(U.sum(axis=1), 1.0)),
            "labels_are_argmax": bool(
                np.array_equal(np.argmax(U, axis=1), np.asarray(opt.labels_))
            ),
            "partition_constrained_declared": type(opt)._partition_constrained,
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", help="comma-separated case names")
    ap.add_argument("--sizes", default="60,120")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    for name in args.cases.split(","):
        case = CASES[name]
        dn, dd, dk = case.default
        for n in (int(s) for s in args.sizes.split(",")):
            for seed in (int(s) for s in args.seeds.split(",")):
                results.append(
                    compare_one(
                        name,
                        n,
                        args.d if args.d is not None else dd,
                        args.k if args.k is not None else dk,
                        seed,
                    )
                )
                r = results[-1]
                mem = r["memberships"]
                detail = (
                    f"max|dU|={mem.get('max_abs'):.3e}"
                    if mem.get("status") == "compared"
                    else mem.get("status")
                )
                print(
                    f"{name:14s} n={n:<5} seed={seed}  {detail}  "
                    f"labels={r.get('label_agreement', float('nan')):.4f}  "
                    f"k {r['n_clusters_ref']}->{r['n_clusters_opt']}",
                    flush=True,
                )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
