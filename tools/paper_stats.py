#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Regenerate every quantitative claim made about SCPP in the paper and README.

Any number describing the size of the library — estimator count, test count,
line coverage, dataset and metric counts — must come from this script rather
than from a manual tally, so that the claims cannot silently drift from the
code.

Usage
-----
    python tools/paper_stats.py                 # counts only (fast)
    python tools/paper_stats.py --with-coverage # also run pytest under coverage

The coverage figure requires the ``dev`` and ``deep`` extras:

    pip install -e ".[dev,deep]"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def count_estimators() -> tuple[int, int]:
    """Return (exported estimator count, algorithm module count)."""
    sys.path.insert(0, str(ROOT))
    import soft_clustering

    modules = [
        p
        for p in (ROOT / "soft_clustering").glob("_*.py")
        if p.name not in ("__init__.py", "_base.py")
    ]
    estimators = [
        n
        for n in soft_clustering.__all__
        if n not in ("BaseSoftClusterer", "DEEP_ESTIMATORS")
    ]
    return len(estimators), len(modules)


def count_tests() -> tuple[int, int]:
    """Return (collected test count, test module count)."""
    modules = list((ROOT / "tests").glob("test_*.py"))
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
    # pytest reports e.g. "532 tests collected in 6.01s"; with -q the same line
    # is preceded by per-file counts, so match the summary explicitly.
    match = re.search(r"(\d+)\s+tests?\s+collected", proc.stdout)
    collected = int(match.group(1)) if match else 0
    if collected == 0:
        raise RuntimeError(
            "Could not parse the collected test count from pytest output:\n"
            + proc.stdout[-2000:]
            + proc.stderr[-2000:]
        )
    return collected, len(modules)


def count_benchmark_assets() -> tuple[int, int, int]:
    """Return (dataset count, soft metric count, hard metric count)."""
    sys.path.insert(0, str(ROOT))
    import numpy as np

    from soft_clustering.benchmarking import metrics
    from soft_clustering.benchmarking.datasets import available_datasets

    n = 40
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    U = rng.random((n, 3))
    U /= U.sum(axis=1, keepdims=True)
    centers = rng.normal(size=(3, 3))
    labels = np.argmax(U, axis=1)

    soft = metrics.soft_clustering_metrics(X, U, centers=centers)
    hard = metrics.clustering_metrics(X, labels, y_true=labels)
    return len(available_datasets()), len(soft), len(hard)


def measure_coverage() -> float:
    """Run the suite under coverage and return the line-coverage percentage."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=soft_clustering",
            "--cov-report=xml:coverage.xml",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,  # keep pytest's output off this script's stdout
        text=True,
    )
    xml = ROOT / "coverage.xml"
    if not xml.exists():
        raise RuntimeError("coverage.xml was not produced; is pytest-cov installed?")
    rate = float(ET.parse(xml).getroot().get("line-rate", "0"))
    return 100.0 * rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-coverage", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable output"
    )
    args = parser.parse_args()

    n_estimators, n_modules = count_estimators()
    n_tests, n_test_modules = count_tests()
    n_datasets, n_soft, n_hard = count_benchmark_assets()

    stats = {
        "estimators_exported": n_estimators,
        "algorithm_modules": n_modules,
        "tests_collected": n_tests,
        "test_modules": n_test_modules,
        "benchmark_datasets": n_datasets,
        "soft_metrics": n_soft,
        "hard_metrics": n_hard,
    }
    if args.with_coverage:
        stats["line_coverage_percent"] = round(measure_coverage(), 1)

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    width = max(len(k) for k in stats)
    print("SCPP quantitative claims (regenerated from source)")
    print("-" * (width + 12))
    for key, value in stats.items():
        print(f"{key.replace('_', ' '):<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
