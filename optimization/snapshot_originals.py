#!/usr/bin/env python3
"""Snapshot the current (pre-optimization) implementations as a reference build.

Creates ``optimization/original/scpp_original/`` — a self-contained, importable
package holding the algorithm modules *verbatim*, plus the unmodified
``_base.py`` they depend on. Nothing here is edited afterwards: it is the
baseline the optimized implementations are measured and verified against.

The snapshot is taken from git's HEAD version of each file when available, so
that re-running this after an optimization still reproduces the true original
rather than the already-optimized file.

Usage
-----
    python optimization/snapshot_originals.py _softdbscangm _mbmm _kmart
    python optimization/snapshot_originals.py --from-git HEAD _softdbscangm
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PKG = ROOT / "soft_clustering"
DEST = Path(__file__).resolve().parent / "original" / "scpp_original"

INIT_TEMPLATE = '''"""Reference (pre-optimization) SCPP implementations.

Verbatim copies of the algorithm modules as they stood before the optimization
study, together with the unmodified ``_base.py`` protocol they rely on. This
package exists so that optimized implementations can be verified and
benchmarked against a stable baseline.

It is deliberately **outside** the production import path: nothing in
``soft_clustering/`` imports from here, and this package is not distributed.

Import it exactly as you would the real package::

    import scpp_original
    model = scpp_original.SoftDBSCANGM()
"""

from ._base import BaseSoftClusterer  # noqa: F401

_ESTIMATORS = {ESTIMATORS!r}
_ALIASES = {ALIASES!r}


def __getattr__(name):
    module_path = _ESTIMATORS.get(name)
    if module_path is None:
        raise AttributeError(f"module {{__name__!r}} has no attribute {{name!r}}")
    from importlib import import_module

    module = import_module(module_path, __name__)
    obj = getattr(module, _ALIASES.get(name, name))
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(list(globals()) + list(_ESTIMATORS))


__all__ = sorted(_ESTIMATORS)
'''


def read_from_git(rel: str, ref: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("modules", nargs="+", help="module stems, e.g. _softdbscangm")
    ap.add_argument(
        "--from-git",
        default=None,
        metavar="REF",
        help="take the file content from this git ref instead of the worktree",
    )
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)

    # _base.py is shared infrastructure, not an optimization target; copy it so
    # the reference package stands alone.
    to_copy = ["_base"] + [m if m.startswith("_") else f"_{m}" for m in args.modules]

    import soft_clustering as sc

    estimators, aliases = {}, {}
    for name, mod in sc._ESTIMATORS.items():
        if mod.lstrip(".") in to_copy:
            estimators[name] = mod
            if name in sc._ALIASES:
                aliases[name] = sc._ALIASES[name]

    for stem in to_copy:
        rel = f"soft_clustering/{stem}.py"
        source = None
        if args.from_git:
            source = read_from_git(rel, args.from_git)
            if source is None:
                print(f"  ! {stem}: not in git {args.from_git}, using worktree")
        if source is None:
            source = (PKG / f"{stem}.py").read_text()
        (DEST / f"{stem}.py").write_text(source)
        print(f"  snapshotted {stem}.py ({len(source.splitlines())} lines)")

    (DEST / "__init__.py").write_text(
        INIT_TEMPLATE.format(ESTIMATORS=estimators, ALIASES=aliases)
    )
    print(f"\nreference package: {DEST}")
    print(f"exposes: {', '.join(sorted(estimators))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
