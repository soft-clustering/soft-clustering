#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Run one estimator configuration in isolation and report JSON on stdout.

Isolating each case in its own process keeps a single pathological algorithm
from stalling a whole sweep, lets the driver impose a per-case timeout, and
gives each measurement a clean allocator and import state.

Usage
-----
    python optimization/run_case.py FCM --n 600 --d 8 --k 3 [--profile] [--module ...]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tracemalloc
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
# The preserved reference build, importable as ``scpp_original`` via --module.
sys.path.insert(0, str(_HERE / "original"))

from harness import CASES, build_inputs, make_estimator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="cProfile the fit")
    ap.add_argument("--memory", action="store_true", help="tracemalloc peak")
    ap.add_argument(
        "--module",
        default="soft_clustering",
        help="import path providing the estimator (e.g. a reference build)",
    )
    args = ap.parse_args()

    case = CASES[args.case]
    dn, dd, dk = case.default
    n = args.n if args.n is not None else dn
    d = args.d if args.d is not None else dd
    k = args.k if args.k is not None else dk

    import importlib
    import time

    module = importlib.import_module(args.module)

    result: dict = {
        "case": args.case,
        "family": case.family,
        "modality": case.modality,
        "module": args.module,
        "n": n,
        "d": d,
        "k": k,
    }

    try:
        build_inputs(case, n, d, k)  # fail fast on an unbuildable configuration

        # Algorithm chatter (several estimators print progress) would corrupt
        # the JSON protocol, so it is captured and discarded.
        sink = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(sink):
            warnings.simplefilter("ignore")

            for _ in range(args.warmup):
                make_estimator(case, k, module).fit(*build_inputs(case, n, d, k))

            times = []
            peak_mb = None
            est = None
            for _ in range(args.repeats):
                est = make_estimator(case, k, module)
                fresh = build_inputs(case, n, d, k)
                if args.memory:
                    tracemalloc.start()
                start = time.perf_counter()
                est.fit(*fresh)
                elapsed = time.perf_counter() - start
                if args.memory:
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_mb = peak / 1024**2
                times.append(elapsed)

            if args.profile:
                import cProfile
                import pstats

                pr = cProfile.Profile()
                pr.enable()
                make_estimator(case, k, module).fit(*build_inputs(case, n, d, k))
                pr.disable()
                st = pstats.Stats(pr)
                st.sort_stats("tottime")
                rows = []
                for func, (_cc, nc, tt, ct, _cal) in list(st.stats.items())[:400]:
                    rows.append(
                        {
                            "file": func[0].rsplit("/", 1)[-1],
                            "line": func[1],
                            "func": func[2],
                            "ncalls": nc,
                            "tottime": tt,
                            "cumtime": ct,
                        }
                    )
                rows.sort(key=lambda r: -r["tottime"])
                result["profile"] = rows[:15]
                result["profile_total"] = st.total_tt

        U = getattr(est, "memberships_", None)
        result.update(
            {
                "status": "ok",
                "times_s": times,
                "fit_time_ms": 1000 * min(times),
                "fit_time_mean_ms": 1000 * (sum(times) / len(times)),
                "peak_traced_mb": peak_mb,
                "membership_shape": None if U is None else list(U.shape),
                "n_clusters": getattr(est, "n_clusters", None),
                "stdout_chars": len(sink.getvalue()),
            }
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            }
        )

    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
