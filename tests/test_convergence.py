# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Stopping rules must not depend on the units of the data.

Several estimators used to stop when the objective changed by less than an
absolute ``tol``. Their objectives scale with the squared data scale, so the
same data expressed in smaller units met the test after a single sweep and
returned an unconverged partition. These tests pin the scale-free behaviour:
for estimators whose model is invariant to a uniform rescaling of ``X``, the
fitted memberships and the number of iterations must not change with it.
"""

from __future__ import annotations

import numpy as np
import pytest

import soft_clustering as sc
from soft_clustering._validation import data_scale, objective_converged


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = ([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    return np.vstack([rng.normal(c, 0.5, (40, 3)) for c in centers])


# Estimators whose model and default hyperparameters are dimensionless, so a
# uniform rescaling of X must leave the partition unchanged. Estimators with a
# parameter in data units (a kernel bandwidth, a DBSCAN radius, ECM's delta,
# a covariance ridge) are deliberately absent.
SCALE_FREE = ["FCM", "PCM", "PFCM", "AFCMSimple", "CAFCM", "RoughKMeans"]


@pytest.mark.parametrize("name", SCALE_FREE)
@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_partition_is_invariant_to_rescaling(name, scale):
    X = _blobs()
    reference = getattr(sc, name)(n_clusters=3, random_state=0).fit(X)
    scaled = getattr(sc, name)(n_clusters=3, random_state=0).fit(X * scale)
    np.testing.assert_allclose(
        scaled.memberships_, reference.memberships_, atol=1e-6, rtol=0
    )
    if hasattr(reference, "n_iter_"):
        assert scaled.n_iter_ == reference.n_iter_


@pytest.mark.parametrize("name", ["FCM", "PCM", "GK", "ECM", "ENTROPYFCM"])
def test_small_scale_data_is_not_stopped_after_one_sweep(name):
    """The regression itself: at 1e-4 scale FCM used to stop after 2 sweeps."""
    X = _blobs()
    kwargs = {"n_clusters": 3, "random_state": 0}
    if name == "ENTROPYFCM":
        # entropy_weight is in squared data units; rescale it with the data.
        kwargs["entropy_weight"] = 1e-8
    if name == "GK":
        kwargs["reg_covar"] = 0.0
    model = getattr(sc, name)(**kwargs).fit(X * 1e-4)
    assert model.n_iter_ > 2


def test_objective_converged():
    assert objective_converged(100.0, 100.0 - 1e-4, 1e-5)
    assert not objective_converged(100.0, 99.0, 1e-5)
    # Invariant to a common rescaling of both values.
    assert objective_converged(1e-8, 1e-8 * (1 - 1e-6), 1e-5)
    assert not objective_converged(1e-8, 0.5e-8, 1e-5)
    assert objective_converged(0.0, 0.0, 1e-5)
    assert not objective_converged(np.inf, 1.0, 1e-5)
    assert not objective_converged(np.nan, 1.0, 1e-5)


def test_data_scale():
    X = _blobs()
    assert data_scale(X * 1e3) == pytest.approx(1e3 * data_scale(X))
    assert data_scale(np.ones((5, 2))) == 1.0
