# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for MBMM (Multivariate Beta Mixture Model)."""

import numpy as np
import pytest
from scipy.special import softmax
from scipy.stats import beta

from soft_clustering import MBMM


@pytest.fixture
def X():
    rng = np.random.default_rng(22)
    return rng.beta(2, 5, (50, 3))


def test_fit_runs(X):
    model = MBMM(n_components=3)
    model.fit(X)
    assert model.resp is not None


def test_predict_proba_shape(X):
    model = MBMM(n_components=3)
    model.fit(X)
    R = model.predict_proba()
    assert R.shape == (50, 3)


def test_responsibilities_sum_to_one(X):
    model = MBMM(n_components=3)
    model.fit(X)
    R = model.predict_proba()
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-6)


def test_predict_labels(X):
    model = MBMM(n_components=3)
    model.fit(X)
    labels = model.predict()
    assert labels.shape == (50,)
    assert set(labels).issubset({0, 1, 2})


def test_alpha_beta_shapes(X):
    model = MBMM(n_components=3)
    model.fit(X)
    assert model.alpha.shape == (3, 3)
    assert model.beta.shape == (3, 3)


def test_k2(X):
    model = MBMM(n_components=2)
    model.fit(X)
    assert model.predict_proba().shape == (50, 2)


@pytest.mark.parametrize(
    "max_iter, tol, expected_updates",
    [(0, 1e-5, 0), (1, 1e-5, 1), (7, 0.0, 7), (30, 1e6, 2)],
    ids=["initial-parameters", "one-update", "iteration-cap", "early-stop"],
)
def test_outputs_use_final_parameters_without_extra_update(
    monkeypatch, max_iter, tol, expected_updates
):
    X = np.random.default_rng(0).uniform(0.01, 0.99, (10, 2))
    updates = []
    original_m_step = MBMM._m_step

    def counted_m_step(self, data):
        updates.append(1)
        original_m_step(self, data)

    monkeypatch.setattr(MBMM, "_m_step", counted_m_step)
    model = MBMM(n_clusters=2, max_iter=max_iter, tol=tol, random_state=0).fit(X)
    assert len(updates) == expected_updates
    log_joint = np.column_stack(
        [
            np.log(model.weights[k] + 1e-10)
            + beta.logpdf(X, model.alpha[k], model.beta[k]).sum(axis=1)
            for k in range(2)
        ]
    )
    expected = softmax(log_joint, axis=1)
    np.testing.assert_allclose(model.resp, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.memberships_, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(model.labels_, expected.argmax(axis=1))
    np.testing.assert_array_equal(model.predict_proba(), model.memberships_)
    np.testing.assert_array_equal(model.predict(), model.labels_)
    for method in (model.predict, model.predict_proba):
        with pytest.raises(NotImplementedError, match="transductive"):
            method(X)
