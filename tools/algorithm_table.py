# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Generate the algorithm table from ``soft_clustering._registry``.

    python tools/algorithm_table.py           # print the Markdown table
    python tools/algorithm_table.py --write   # update README.md in place

The README table lives between ``<!-- algorithms:start -->`` and
``<!-- algorithms:end -->``. ``tests/test_registry.py`` fails when it is out of
date, so the table cannot drift from the registry.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from soft_clustering import DEEP_ESTIMATORS
from soft_clustering._registry import ALGORITHMS, OUTPUT_KINDS, STATUS

ROOT = Path(__file__).resolve().parent.parent

#: Display order of the groups, keyed by the first word of the family.
GROUPS = [
    ("Fuzzy and kernel", ("fuzzy", "kernel")),
    ("Possibilistic and evidential", ("possibilistic", "evidential")),
    ("Probabilistic, density and rough-set", ("probabilistic", "density", "rough")),
    ("Graphs", ("graph",)),
    ("Documents", ("topic", "document")),
    ("Ensembles", ("ensemble",)),
]


def _group(family: str) -> str:
    head = family.split(",")[0].strip().split()[0]
    for title, keys in GROUPS:
        if head.startswith(keys):
            return title
    raise ValueError(f"no table group for family {family!r}")


def markdown_table() -> str:
    lines = [
        "| Estimator | Method | Input | Output | K | Status | Reference |",
        "| --- | --- | --- | :---: | :---: | :---: | --- |",
    ]
    for title, _ in GROUPS:
        lines.append(f"| **{title}** | | | | | | |")
        members = [a for a in ALGORITHMS.values() if _group(a.family) == title]
        for a in sorted(members, key=lambda a: a.name.lower()):
            name = f"`{a.name}`" + (" ⚡" if a.name in DEEP_ESTIMATORS else "")
            ref = f"[{a.reference}]({a.link})" if a.link else a.reference
            lines.append(
                f"| {name} | {a.method} | {a.input} | {a.output} | "
                f"{a.n_clusters} | {a.status} | {ref} |"
            )
    lines.append("")
    lines.append(
        "**Output** (what `memberships_` holds): "
        + "; ".join(f"**{k}** {v}" for k, v in OUTPUT_KINDS.items())
        + "."
    )
    lines.append("")
    lines.append(
        "**Status**: "
        + "; ".join(f"**{k}** — {v}" for k, v in STATUS.items())
        + ". ⚡ requires the `deep` extra."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update README.md")
    args = parser.parse_args()
    table = markdown_table()
    if not args.write:
        print(table, end="")
        return
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(<!-- algorithms:start -->\n).*?(<!-- algorithms:end -->)", re.S
    )
    if not pattern.search(text):
        raise SystemExit("README.md has no <!-- algorithms:start/end --> markers")
    readme.write_text(
        pattern.sub(lambda m: m.group(1) + table + m.group(2), text), encoding="utf-8"
    )
    print(f"updated {readme}")


if __name__ == "__main__":
    main()
