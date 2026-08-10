#!/usr/bin/env python3
"""Drive ``run_case.py`` across many configurations, one subprocess each.

Each case gets its own process and its own timeout, so a single pathological
algorithm cannot stall the sweep — and a timeout is recorded as a result rather
than lost, because "does not finish at this size" is itself a measurement.

Results stream to a JSONL file as they complete, so a long sweep is resumable
and partial results survive an interruption.

Usage
-----
    python optimization/sweep.py --out raw/smoke.jsonl --n 200 --timeout 60
    python optimization/sweep.py --out raw/scale.jsonl --sizes 250,500,1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import CASES  # noqa: E402


def run(case: str, *, n, d, k, timeout, extra, python) -> dict:
    cmd = [python, str(HERE / "run_case.py"), case]
    for flag, value in (("--n", n), ("--d", d), ("--k", k)):
        if value is not None:
            cmd += [flag, str(value)]
    cmd += extra
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=HERE.parent
        )
    except subprocess.TimeoutExpired:
        return {
            "case": case,
            "n": n,
            "d": d,
            "k": k,
            "status": "timeout",
            "timeout_s": timeout,
        }
    wall = time.perf_counter() - started
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        result = json.loads(line)
    except json.JSONDecodeError:
        return {
            "case": case,
            "n": n,
            "d": d,
            "k": k,
            "status": "crash",
            "returncode": proc.returncode,
            "stderr": proc.stderr[-400:],
        }
    result["wall_s"] = wall
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="", help="comma-separated subset")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--sizes", default="", help="comma-separated n values to sweep")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--module", default=None)
    args = ap.parse_args()

    extra: list[str] = []
    if args.profile:
        extra.append("--profile")
    if args.memory:
        extra.append("--memory")
    if args.repeats != 1:
        extra += ["--repeats", str(args.repeats)]
    if args.warmup:
        extra += ["--warmup", str(args.warmup)]
    if args.module:
        extra += ["--module", args.module]

    names = [c for c in args.cases.split(",") if c] or list(CASES)
    sizes = [int(s) for s in args.sizes.split(",") if s] or [args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = len(names) * len(sizes)
    done = 0
    with out.open("w") as fh:
        for n in sizes:
            for name in names:
                done += 1
                result = run(
                    name,
                    n=n,
                    d=args.d,
                    k=args.k,
                    timeout=args.timeout,
                    extra=extra,
                    python=args.python,
                )
                fh.write(json.dumps(result) + "\n")
                fh.flush()
                status = result.get("status")
                ms = result.get("fit_time_ms")
                detail = f"{ms:9.1f} ms" if ms is not None else status
                print(
                    f"[{done:3d}/{total}] {name:14s} n={n or '-':<6} {status:8s} {detail}",
                    flush=True,
                )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
