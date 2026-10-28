# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""scikit-learn's own estimator checks, run on the feature-matrix estimators.

This is the evidence behind the documentation's compatibility statement. The
checks run on the inductive estimators that take a plain feature matrix; the
others take graphs, documents or several inputs and are outside what
``check_estimator`` can exercise.

Two checks are expected to fail, for one reason: scikit-learn's sparse-input
checks assume that an estimator with ``predict_proba`` is a classifier and
read its classifier tags. SCPP uses ``predict_proba`` for membership
matrices, as scikit-learn's own ``GaussianMixture`` does. Fitting and
predicting on sparse input works (``tests/test_protocol.py``); only the
shape assertion of the check does not apply.
"""

from __future__ import annotations

import pytest

sklearn_checks = pytest.importorskip("sklearn.utils.estimator_checks")
if not hasattr(sklearn_checks, "parametrize_with_checks"):  # pragma: no cover
    pytest.skip("needs scikit-learn's parametrize_with_checks", allow_module_level=True)

import soft_clustering as sc  # noqa: E402

EXPECTED_FAILURES = {
    "check_estimator_sparse_array": (
        "the check treats predict_proba as a classifier method; SCPP's returns "
        "memberships"
    ),
    "check_estimator_sparse_matrix": (
        "the check treats predict_proba as a classifier method; SCPP's returns "
        "memberships"
    ),
}

ESTIMATORS = [
    sc.FCM(n_clusters=3, random_state=0),
    sc.GK(n_clusters=3, random_state=0),
    sc.GMM(n_clusters=3, random_state=0),
    sc.PCM(n_clusters=3, random_state=0),
    sc.PFCM(n_clusters=3, random_state=0),
    sc.GathGeva(n_clusters=3, random_state=0),
    sc.ECM(n_clusters=3, random_state=0),
    sc.ENTROPYFCM(n_clusters=3, random_state=0),
    sc.AFCMSimple(n_clusters=3, random_state=0),
    sc.KFCM(n_clusters=3, random_state=0),
    sc.FCC(n_clusters=3, random_state=0),
]


@sklearn_checks.parametrize_with_checks(
    ESTIMATORS, expected_failed_checks=lambda estimator: EXPECTED_FAILURES
)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
