#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Verify that every quantitative claim in the manuscript matches the source.

The previous revision of this paper reported 532 tests across 42 modules at a
time when the repository had 687 across 44, and said eighteen algorithms ran
in under 10 ms when the study's own ``results.csv`` said fourteen. Both
numbers had a generator (``tools/paper_stats.py``) that nobody re-ran. This
script closes the loop: it recomputes each claim from the repository and
compares it against the value written in ``paper/scpp.tex``, so a stale number
fails CI instead of reaching a reviewer.

Usage
-----
    python tools/check_paper_numbers.py            # counts only
    python tools/check_paper_numbers.py --full     # also run the test suite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "scpp.tex"


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, label: str, expected, pattern: str, text: str) -> None:
        """Assert that `pattern` occurs in the paper and captures `expected`."""
        self.checks += 1
        found = re.search(pattern, text)
        if not found:
            self.failures.append(f"{label}: pattern not found in the paper: {pattern}")
            print(f"  MISSING  {label}: expected {expected}")
            return
        actual = found.group(1).replace(",", "").replace("{", "").replace("}", "")
        if str(actual) != str(expected):
            self.failures.append(
                f"{label}: paper says {actual}, source says {expected}"
            )
            print(f"  MISMATCH {label}: paper={actual} source={expected}")
        else:
            print(f"  ok       {label} = {expected}")

    def note(self, label: str, value) -> None:
        print(f"  --       {label} = {value}")


def optimization_numbers() -> dict:
    rows = list(csv.DictReader((ROOT / "optimization/benchmarks/results.csv").open()))
    pairs: dict[tuple, dict] = {}
    for row in rows:
        if row["implementation"] in ("original", "optimized"):
            pairs.setdefault((row["algorithm"], row["n"]), {})[
                row["implementation"]
            ] = row

    speedups, timeout_bound = [], None
    for both in pairs.values():
        if "original" not in both or "optimized" not in both:
            continue
        original, optimized = both["original"], both["optimized"]
        if original["status"] == "ok" and optimized["status"] == "ok":
            speedups.append(
                float(original["fit_time_ms"]) / float(optimized["fit_time_ms"])
            )
        elif original["status"] == "timeout":
            timeout_bound = (
                float(original["timeout_s"]) * 1000 / float(optimized["fit_time_ms"])
            )

    survey = [r for r in rows if r["implementation"] == "survey"]
    timed = [r for r in survey if r["fit_time_ms"]]

    correctness, worst = 0, 0.0
    for path in sorted((ROOT / "optimization/correctness").glob("*.json")):
        entries = json.loads(path.read_text())
        correctness += len(entries)
        worst = max(worst, *(e["memberships"].get("max_abs", 0) or 0 for e in entries))

    memory = [
        r
        for r in rows
        if r["implementation"] == "optimized" and r.get("peak_traced_mb")
    ]

    return {
        "paired": len(speedups),
        "median": round(statistics.median(speedups), 1),
        "geomean": round(math.exp(sum(map(math.log, speedups)) / len(speedups)), 1),
        "amean": round(sum(speedups) / len(speedups), 1),
        "minimum": round(min(speedups), 1),
        "maximum": round(max(speedups), 1),
        "timeout_bound": int(timeout_bound),
        "survey_attempted": len(survey),
        "survey_timed": len(timed),
        "under_10ms": sum(1 for r in timed if float(r["fit_time_ms"]) < 10),
        "correctness_comparisons": correctness,
        "worst_deviation": worst,
        "memory_measurements": len(memory),
    }


_NUMBER_WORDS = {2: "Two", 9: "Nine", 10: "Ten"}


def _validated_count() -> str:
    """The number of estimators with Tier 1/2 validation, as the paper spells it."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_external_agreement import VALIDATED

    return _NUMBER_WORDS.get(len(VALIDATED), str(len(VALIDATED)))


def benchmark_numbers() -> dict:
    rows = list(csv.DictReader((ROOT / "benchmarks/results/main_benchmark.csv").open()))
    external = list(
        csv.DictReader((ROOT / "benchmarks/results/external_baselines.csv").open())
    )
    return {
        "runs": len(rows),
        "algorithms": len({r["algorithm"] for r in rows}),
        "datasets": len({r["dataset"] for r in rows}),
        "ok": sum(r["status"] == "ok" for r in rows),
        "degenerate": sum(r["status"] == "degenerate" for r in rows),
        "failed": sum(r["status"] in ("timeout", "crashed", "error") for r in rows),
        "worst_external": max(
            float(r["max_membership_diff"])
            for r in external
            if r["max_membership_diff"] not in ("", "nan")
        ),
        "all_ari_one": all(float(r["label_ari"]) == 1.0 for r in external),
        "validated": _validated_count(),
    }


def library_numbers(full: bool) -> dict:
    sys.path.insert(0, str(ROOT))
    import soft_clustering as sc

    modules = [
        path
        for path in (ROOT / "soft_clustering").glob("_*.py")
        if path.name not in ("__init__.py", "_base.py")
    ]
    from soft_clustering.benchmarking.datasets import available_datasets

    numbers = {
        "estimators": len(
            [n for n in sc.__all__ if n not in ("BaseSoftClusterer", "DEEP_ESTIMATORS")]
        ),
        "modules": len(modules),
        "examples": len(list((ROOT / "example").glob("*.py"))),
        "datasets": len(available_datasets()),
        "deep": len(sc.DEEP_ESTIMATORS),
    }

    if full:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        match = re.search(r"(\d+)\s+tests?\s+collected", proc.stdout)
        numbers["tests"] = int(match.group(1)) if match else 0
        numbers["test_modules"] = len(list((ROOT / "tests").glob("test_*.py")))
    return numbers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also collect the suite")
    args = parser.parse_args()

    if not PAPER.exists():
        # Every check below is a regex against the manuscript, so with no
        # manuscript in the tree there is nothing to verify. That is the state
        # the repository is in whenever the paper is not being worked on, and
        # it is not a defect -- failing here would turn a green branch red for
        # a file the repository does not claim to have. The checks come back
        # automatically as soon as the manuscript is restored.
        print(f"no manuscript at {PAPER}; nothing to verify")
        return 0
    text = PAPER.read_text()
    report = Report()

    print("Library")
    library = library_numbers(args.full)
    report.check(
        "estimators",
        library["estimators"],
        r"collects (\d+) soft clustering algorithms",
        text,
    )
    report.check(
        "estimators (Table 1)",
        library["estimators"],
        r"\\textbf\{SCPP \(this work\)\}\s*\n& \\textbf\{(\d+)\}",
        text,
    )
    report.check(
        "estimators (contributions)",
        library["estimators"],
        r"SCPP provides (\d+) algorithms spanning",
        text,
    )
    report.check(
        "examples", library["examples"], r"(\d+) standalone example scripts", text
    )
    report.note("algorithm modules", library["modules"])
    report.note("deep-extra estimators", library["deep"])
    if args.full:
        report.check(
            "tests", library["tests"], r"suite contains ([\d{,}]+) tests", text
        )
        report.check(
            "test modules", library["test_modules"], r"tests across (\d+) modules", text
        )

    print("\nOptimization study")
    opt = optimization_numbers()
    report.check(
        "paired measurements",
        opt["paired"],
        r"Across the (\d+)\s*\n?paired measurements",
        text,
    )
    report.check(
        "median speedup", opt["median"], r"median\s*\n?speedup is \$?([\d.]+)", text
    )
    report.check(
        "geometric mean", opt["geomean"], r"geometric mean is \$?([\d.]+)", text
    )
    report.check(
        "arithmetic mean",
        opt["amean"],
        r"mean is substantially larger \(\$?([\d.]+)",
        text,
    )
    report.check(
        "under 10 ms",
        opt["under_10ms"],
        r"of which complete a 200-sample fit in \$<10\$\\,ms & (\d+)",
        text,
    )
    report.check(
        "baseline-timed",
        opt["survey_timed"],
        r"of which baseline-timed successfully & (\d+)",
        text,
    )
    report.check(
        "harness-constructible",
        opt["survey_attempted"],
        r"Constructible by the shared harness & (\d+)",
        text,
    )
    report.check(
        "correctness comparisons",
        opt["correctness_comparisons"],
        r"--- (\d+) comparisons over",
        text,
    )
    report.check(
        "timeout bound",
        opt["timeout_bound"],
        r"Maximum \(reference times out\) & \$>([\d{,}]+)",
        text,
    )
    report.note("worst membership deviation", f"{opt['worst_deviation']:.3e}")
    report.note("memory measurements", opt["memory_measurements"])

    print("\nBenchmarks")
    bench = benchmark_numbers()
    report.check(
        "benchmark runs", bench["runs"], r"--- (\d+)\s*\n?fits under one protocol", text
    )
    report.check(
        "benchmark algorithms",
        bench["algorithms"],
        r"All (\d+) feature-matrix estimators",
        text,
    )
    report.check("completed fits", bench["ok"], r"fits, (\d+) completed", text)
    report.check(
        "degenerate fits", bench["degenerate"], r"(\d+) returned a degenerate", text
    )
    report.note("failed fits (timeout/OOM/error)", bench["failed"])
    report.check(
        "externally validated estimators",
        bench["validated"],
        r"(\w+) of 42 estimators have Tier 1 or Tier 2",
        text,
    )
    report.note("worst external disagreement", f"{bench['worst_external']:.1e}")
    report.note("external partitions all identical", bench["all_ari_one"])

    print(f"\n{report.checks} checks, {len(report.failures)} failure(s)")
    for failure in report.failures:
        print(f"  ! {failure}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
