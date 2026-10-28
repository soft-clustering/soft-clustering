# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Every script in ``example/`` must run against the current API.

Before this test existed, 14 of the 42 examples had stopped working without
anyone noticing.
"""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "example").glob("*.py"))
DEEP = {"example_cdcgs.py", "example_dmon.py", "example_nocd.py", "example_rdfkc.py"}
HAVE_DEEP = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not EXAMPLES, reason="example/ is not shipped with this install")
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(path, capsys):
    if path.name in DEEP and not HAVE_DEEP:
        pytest.skip("needs the optional `deep` extra")
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out.strip(), f"{path.name} printed nothing"
