# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Behavioral checks for the benchmark scripts, without fitting a benchmark suite."""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


main_benchmark = _load_module("benchmarks/run_main_benchmark.py")


@pytest.mark.parametrize(
    "module_path",
    ["benchmarks/run_external_baselines.py", "tests/test_external_agreement.py"],
)
def test_alignment_is_bijective_and_preserves_equivalent_partitions(module_path):
    align = _load_module(module_path).align
    for reference, order in (
        (np.array([[0.1, 0.9], [0.2, 0.8]]), [0, 1]),
        (np.array([[0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]), [2, 0, 1]),
        (np.array([[0.2, 0.2, 0.6], [0.1, 0.1, 0.8]]), [2, 1, 0]),
    ):
        candidate = reference[:, order].copy()
        original = candidate.copy()
        expected = reference.copy()
        actual = align(reference, candidate)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(candidate, original)
        np.testing.assert_array_equal(reference, expected)
        assert actual.shape == reference.shape
    with pytest.raises(ValueError, match="equally shaped"):
        align(np.ones((2, 2)), np.ones((2, 3)))


def test_summary_accounts_for_every_status_and_only_successful_finite_values():
    statuses = [
        "ok",
        "slow",
        "degenerate",
        "error",
        "construct_error",
        "timeout",
        "crashed",
        "empty_partition",
        "metric_error",
        "future_failure",
        "ok",
    ]
    columns = ("ari", "nmi", "silhouette", "fit_time_ms", "peak_traced_mb")
    rows = []
    for i, status in enumerate(statuses):
        value = {0: 2.0, 1: 4.0, 10: np.nan}.get(i, 999.0)
        rows.append(
            dict(
                algorithm="A",
                dataset=f"data_{i % 2}",
                status=status,
                **dict.fromkeys(columns, value),
            )
        )
    rows.insert(
        0,
        dict(
            algorithm="B", dataset="other", status="ok", **dict.fromkeys(columns, 8.0)
        ),
    )
    first, second = main_benchmark.summarise(rows)
    assert first["algorithm"] == "A" and second["algorithm"] == "B"
    assert (
        first["datasets"],
        first["runs"],
        first["completed"],
        first["degenerate"],
    ) == (2, 11, 3, 1)
    breakdown = (
        "errors",
        "timeouts",
        "crashes",
        "empty_partitions",
        "metric_errors",
        "other_failures",
    )
    assert [first[key] for key in breakdown] == [2, 1, 1, 1, 1, 1]
    for entry in (first, second):
        assert entry["failures"] == sum(entry[key] for key in breakdown)
        assert (
            entry["runs"]
            == entry["completed"] + entry["degenerate"] + entry["failures"]
        )
    assert (
        second["datasets"],
        second["runs"],
        second["completed"],
        second["failures"],
    ) == (1, 1, 1, 0)
    for column in columns:
        assert first[f"mean_{column}"] == pytest.approx(3.0)
        assert second[f"mean_{column}"] == pytest.approx(8.0)


def test_summary_table_displays_aggregate_failures(tmp_path, monkeypatch):
    row = dict(
        algorithm="Example",
        dataset="data",
        seed=0,
        status="timeout",
        ari=np.nan,
        nmi=np.nan,
        silhouette=np.nan,
        fit_time_ms=np.nan,
        peak_traced_mb=np.nan,
    )
    monkeypatch.setattr(main_benchmark, "OUT", tmp_path)
    main_benchmark.write_outputs([row], main_benchmark.summarise([row]), [], [])
    table = (tmp_path / "main_benchmark.tex").read_text()
    assert "Degen. & Failures" in table
    assert "Example & 1 & 0 & 0 & 1 &" in table
    with (tmp_path / "main_benchmark_summary.csv").open(newline="") as handle:
        entry = next(csv.DictReader(handle))
    assert entry["errors"] == "0" and entry["timeouts"] == entry["failures"] == "1"
