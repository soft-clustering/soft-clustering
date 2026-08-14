#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Static audit of every SCPP algorithm module.

Emits measured facts only — line counts, loop nesting, and the specific
numerical anti-patterns that profiling later confirms or refutes. Nothing here
is a judgement about performance; it is the shortlist that Phase 2 measures.

Usage
-----
    python optimization/audit_static.py            # markdown table to stdout
    python optimization/audit_static.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "soft_clustering"

# Numerical anti-patterns worth flagging for profiling. Each maps to a short
# code used in the inventory table.
PATTERNS = {
    "inv": ("np.linalg.inv", "explicit matrix inverse (prefer solve)"),
    "pinv": ("np.linalg.pinv", "pseudo-inverse in a loop is costly"),
    "det": ("np.linalg.det", "determinant per cluster per iteration"),
    "eig": ("np.linalg.eig", "dense eigendecomposition"),
    "svd": ("np.linalg.svd", "dense SVD"),
    "norm": ("np.linalg.norm", "often recomputable via cdist/einsum"),
    "cdist": ("cdist", "already delegated to SciPy"),
    "einsum": ("einsum", "already vectorised"),
    "append": (".append(", "list building inside numerical code"),
}


class ModuleAudit(ast.NodeVisitor):
    """Collect loop structure and call sites from one algorithm module."""

    def __init__(self) -> None:
        self.max_loop_depth = 0
        self.n_loops = 0
        self.loops_with_calls = 0
        self._depth = 0
        self.calls: dict[str, int] = {}
        self.comprehensions = 0
        self.classes: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)

    def _visit_loop(self, node) -> None:
        self.n_loops += 1
        self._depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self._depth)
        # A loop whose body calls into numpy is the classic vectorisation target
        if any(isinstance(n, ast.Call) for n in ast.walk(node)):
            self.loops_with_calls += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_For = _visit_loop
    visit_While = _visit_loop

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        for key, (needle, _) in PATTERNS.items():
            if needle.strip("(.") in name:
                self.calls[key] = self.calls.get(key, 0) + 1
        self.generic_visit(node)


def audit_module(path: Path) -> dict:
    source = path.read_text()
    tree = ast.parse(source)
    visitor = ModuleAudit()
    visitor.visit(tree)

    code_lines = [
        ln
        for ln in source.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return {
        "module": path.name,
        "loc": len(source.splitlines()),
        "code_loc": len(code_lines),
        "classes": visitor.classes,
        "n_loops": visitor.n_loops,
        "max_loop_depth": visitor.max_loop_depth,
        "loops_with_calls": visitor.loops_with_calls,
        "comprehensions": visitor.comprehensions,
        "patterns": visitor.calls,
        "uses_torch": "import torch" in source,
        "uses_scipy_sparse": "scipy.sparse" in source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    modules = sorted(
        p for p in PKG.glob("_*.py") if p.name not in ("__init__.py", "_base.py")
    )
    rows = [audit_module(p) for p in modules]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    header = (
        "| Module | Code LOC | Loops | Max depth | Loops w/ calls | Comprehensions | "
        "Flags | Torch |"
    )
    print(header)
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | :-: |")
    for r in rows:
        flags = ", ".join(f"{k}x{v}" for k, v in sorted(r["patterns"].items())) or "—"
        print(
            f"| `{r['module']}` | {r['code_loc']} | {r['n_loops']} | "
            f"{r['max_loop_depth']} | {r['loops_with_calls']} | "
            f"{r['comprehensions']} | {flags} | "
            f"{'yes' if r['uses_torch'] else ''} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
