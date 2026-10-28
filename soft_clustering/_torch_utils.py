# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Random-state isolation for the PyTorch-backed estimators.

Only imported by modules that already require torch.
"""

from __future__ import annotations

import contextlib

import torch

from ._validation import int_seed


@contextlib.contextmanager
def isolated_torch_rng(random_state):
    """Run a block with a private torch random stream.

    The global torch RNG state is saved on entry and restored on exit
    (``torch.random.fork_rng``), so fitting an estimator never perturbs the
    caller's random stream. When ``random_state`` is not ``None`` the private
    stream is seeded with it, which makes the block reproducible; earlier
    releases called ``torch.manual_seed`` directly and so reseeded the
    caller's global generator as a side effect.
    """
    seed = int_seed(random_state)
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(seed)
        yield
