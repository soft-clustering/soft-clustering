# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Fits must not depend on the interpreter's string-hash seed.

Python randomises ``hash(str)`` per process (``PYTHONHASHSEED``), so iterating
over a ``set`` or ``dict`` of words visits them in a different order in every
new process. An estimator whose result depends on that order is reproducible
within one process -- which is all ``tests/test_protocol.py`` can check -- and
not across runs of the same script. WBSC had exactly this defect. The test
fits each document estimator in subprocesses with different hash seeds and
compares the results.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = r"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import soft_clustering as sc

rng = np.random.default_rng(5)
words = ("fuzzy clustering membership degree centroid distance kernel graph "
         "community detection network overlapping topic document word mixture "
         "gaussian soft assignment spectral possibilistic evidential").split()
docs = [" ".join(rng.choice(words, size=8)) for _ in range(80)]
name = sys.argv[2]
cls = getattr(sc, name)
kwargs = {"random_state": 0} if "random_state" in cls().get_params() else {}
if name in ("LDA", "PLSI"):
    kwargs["n_clusters"] = 3
U = np.asarray(cls(**kwargs).fit(docs).memberships_, dtype=float)
print(U.shape, np.round(U, 8).tobytes().hex())
"""


def _fit_in_subprocess(name: str, hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(ROOT), name],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout


@pytest.mark.parametrize("name", ["WBSC", "SISC", "KMART", "LDA", "PLSI"])
def test_document_estimators_do_not_depend_on_the_hash_seed(name):
    outputs = {_fit_in_subprocess(name, seed) for seed in ("1", "2", "3")}
    assert len(outputs) == 1, f"{name} gave different results across processes"
