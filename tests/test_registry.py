# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""The algorithm registry must describe the code, not an idea of it.

``soft_clustering/_registry.py`` is the single source of the algorithm tables
in the README and the documentation. These tests bind its claims to the
estimator classes, so a table cannot advertise a property the code lacks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import soft_clustering as sc
from soft_clustering._registry import (
    ALGORITHMS,
    CLUSTER_COUNT,
    OUTPUT_KINDS,
    STATUS,
    list_algorithms,
)

ROOT = Path(__file__).resolve().parent.parent

#: Output kinds whose rows sum to one by definition.
NORMALISED = {"F", "P", "E", "M", "S", "H"}


def test_registry_covers_exactly_the_exported_estimators():
    assert set(ALGORITHMS) == set(sc._ESTIMATORS)


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_coded_fields_are_valid(name):
    info = ALGORITHMS[name]
    assert info.output in OUTPUT_KINDS
    assert info.n_clusters in CLUSTER_COUNT
    assert info.status in STATUS
    assert info.reference
    if info.link:
        assert info.link.startswith("https://")


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_output_kind_matches_the_partition_constraint(name):
    """A normalised output kind must not be declared unconstrained, and
    vice versa (KMART is conditionally normalised, see test_protocol.py)."""
    if name in sc.DEEP_ESTIMATORS:
        pytest.importorskip("torch")
    cls = getattr(sc, name)
    info = ALGORITHMS[name]
    if info.output in NORMALISED:
        assert cls._partition_constrained, f"{name}: output {info.output} sums to one"
    elif info.output in {"T", "A", "D"}:
        assert (
            not cls._partition_constrained
        ), f"{name}: output {info.output} is unnormalised"


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_discovered_cluster_count_matches_the_class(name):
    if name in sc.DEEP_ESTIMATORS:
        pytest.importorskip("torch")
    discovered = ALGORITHMS[name].n_clusters == "discovered"
    assert getattr(sc, name)._determines_k == discovered


def test_validated_estimators_have_an_external_check():
    """Every 'validated' claim is backed by tests/test_external_agreement.py."""
    from tests.test_external_agreement import VALIDATED

    claimed = {a.name for a in list_algorithms(status="validated")}
    assert claimed <= set(VALIDATED), sorted(claimed - set(VALIDATED))


def test_list_algorithms_filters():
    assert [a.name for a in list_algorithms(family="evidential")] == ["ECM", "EVCLUS"]
    assert all(a.status == "unverified" for a in list_algorithms(status="unverified"))
    with pytest.raises(ValueError):
        list_algorithms(status="perfect")


def test_readme_table_is_generated_from_the_registry():
    """The README table is produced by tools/algorithm_table.py; regenerate it
    with ``python tools/algorithm_table.py --write`` after editing the
    registry."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "algorithm_table", ROOT / "tools" / "algorithm_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(
        r"<!-- algorithms:start -->\n(.*?)<!-- algorithms:end -->", readme, re.S
    )
    assert block, "README.md has no generated algorithm table"
    assert (
        block.group(1) == module.markdown_table()
    ), "README algorithm table is stale; run `python tools/algorithm_table.py --write`"
