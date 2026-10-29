# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Unit and integration tests for PLSI (Probabilistic Latent Semantic Indexing)."""

import numpy as np
import pytest
import scipy.sparse as sp

from soft_clustering import PLSI
from soft_clustering._plsi import _split_tokens


@pytest.fixture
def docs():
    return [
        "machine learning data cluster algorithm",
        "fuzzy clustering membership centroid",
        "neural network deep learning training",
        "graph community detection network nodes",
        "topic model latent word document text",
        "probabilistic mixture gaussian expectation",
        "soft assignment cluster membership degree",
        "kernel distance similarity metric function",
    ]


def test_fit_predict_on_text_runs(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_w_given_z is not None


def test_P_w_given_z_shape(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_w_given_z.shape[0] == 3


def test_P_d_given_z_shape(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_d_given_z.shape == (3, 8)


def test_P_z_shape(docs):
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(docs)
    assert model.P_z.shape == (3,)
    np.testing.assert_allclose(model.P_z.sum(), 1.0, atol=1e-5)


def test_fit_predict_on_sparse():
    rng = np.random.default_rng(36)
    X = sp.csr_matrix(rng.integers(0, 5, (10, 15)).astype(float))
    model = PLSI(n_components=3, max_iter=5)
    model.fit_predict(X)
    assert model.P_w_given_z is not None


def test_k2(docs):
    model = PLSI(n_components=2, max_iter=5)
    model.fit_predict(docs)
    assert model.P_d_given_z.shape == (2, 8)


def test_vocab_set_on_text(docs):
    model = PLSI(n_components=2, max_iter=5)
    model.fit_predict(docs)
    assert model.vocab is not None


def test_square_posterior_has_canonical_orientation():
    model = PLSI(n_topics=2, max_iter=1, n_init=1, random_state=0)
    assert model.fit(["cats cats cats dogs", "dogs cars"]) is model
    assert model.memberships_.shape == (2, 2)
    assert np.isfinite(model.memberships_).all()
    np.testing.assert_allclose(model.memberships_.sum(axis=1), 1)
    np.testing.assert_array_equal(model.memberships_, model.P_z_given_d.T)
    assert model.labels_.shape == (2,)
    assert (
        model.n_clusters_ == model.n_clusters == model.get_params()["n_clusters"] == 2
    )


@pytest.mark.parametrize(
    "data",
    [
        ["cats dogs", "cars roads", "flowers trees"],
        sp.csr_matrix([[2, 1, 0], [0, 3, 1], [1, 0, 4]]),
    ],
)
def test_keyword_and_positional_canonical_and_legacy_arrays_agree(data):
    first = PLSI(n_clusters=2, max_iter=2, n_init=1, random_state=0)
    second = PLSI(n_clusters=2, max_iter=2, n_init=1, random_state=0)
    result = first.fit_predict(data)
    keyword = second.fit_predict(data=data)
    assert result is first.memberships_
    assert keyword is second.memberships_
    assert result.shape == (3, 2)
    for attr in (
        "memberships_",
        "labels_",
        "P_z_given_d",
        "P_d_given_z",
        "P_w_given_z",
    ):
        np.testing.assert_array_equal(getattr(first, attr), getattr(second, attr))
    assert first.P_z_given_d.shape == first.P_d_given_z.shape == (2, 3)
    assert first.P_w_given_z.shape[0] == 2
    words = first.get_P_w_given_d()
    assert words.shape == (3, first.P_w_given_z.shape[1])
    np.testing.assert_allclose(words, first.memberships_ @ first.P_w_given_z)


@pytest.mark.parametrize(
    "data",
    [
        ["cats dogs", ""],
        ["cats dogs", "a !"],
        sp.csr_matrix([[1, 2], [0, 0]]),
        np.zeros((2, 2)),
    ],
)
def test_empty_documents_are_rejected(data):
    with pytest.raises(ValueError, match="PLSI: empty documents.*observed tokens"):
        PLSI(n_clusters=2, max_iter=2, n_init=1).fit(data)


def test_all_empty_text_is_rejected():
    with pytest.raises(ValueError, match="PLSI:.*no observed word tokens"):
        PLSI(n_clusters=2).fit(["", "a !"])


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_counts_are_rejected(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        PLSI(n_clusters=2).fit(sp.csr_matrix([[1, bad], [2, 1]]))


def test_tem_repeated_tokens_and_repeat_fit_are_finite_and_reproducible():
    data = ["cats " * 100, "cars " * 100, "flowers " * 100]
    model = PLSI(n_clusters=2, n_init=2, max_iter=10, random_state=7, tempered=True)
    first = model.fit_predict(data).copy()
    diagnostics = (model.perplexity, model.heldout_log_likelihood_, model.beta_)
    history = model.log_likelihoods.copy()
    assert np.isfinite(first).all() and (first >= 0).all()
    np.testing.assert_allclose(first.sum(1), 1)
    assert np.isfinite(diagnostics).all() and np.isfinite(history).all()
    np.testing.assert_array_equal(model.fit_predict(data), first)
    assert (model.perplexity, model.heldout_log_likelihood_, model.beta_) == diagnostics
    assert model.log_likelihoods == history


@pytest.mark.parametrize("draw", ["random", "all", "none"])
def test_token_split_conserves_counts_and_support(draw):
    X = sp.csr_matrix([[1, 0, 0], [0, 20, 30]], dtype=float)

    class ForcedDraw:
        def binomial(self, counts, ratio):
            return counts.copy() if draw == "all" else np.zeros_like(counts)

        def choice(self, positions, p):
            return positions[0]

    def split():
        rng = np.random.default_rng(7) if draw == "random" else ForcedDraw()
        return _split_tokens(X, rng, 0.2)

    parts = split()
    for first, repeated in zip(parts, split()):
        for values, again in zip(first, repeated):
            np.testing.assert_array_equal(values, again)
    train, heldout = [sp.csr_matrix((c, (d, w)), shape=X.shape) for d, w, c in parts]
    np.testing.assert_array_equal((train + heldout).toarray(), X.toarray())
    assert (train.data >= 0).all() and (heldout.data >= 0).all()
    assert np.all(np.asarray(train.sum(1)) > 0)
    assert heldout.sum() > 0
    if draw == "random":
        # A repeated word's occurrences can be split across both sets.
        assert train[1, 1] > 0 and heldout[1, 1] > 0


def test_tem_impossible_singleton_corpus_rejects():
    with pytest.raises(ValueError, match="nonempty TEM holdout.*two tokens"):
        PLSI(n_clusters=2, tempered=True, random_state=0).fit(sp.eye(3, format="csr"))


def test_plain_em_accepts_fractional_weights_but_tem_does_not():
    X = sp.csr_matrix([[0.5, 1.2], [2.5, 0.7], [1.1, 4.2]])
    model = PLSI(
        n_clusters=2, n_init=1, max_iter=10, random_state=0, heldout_ratio=0.0
    ).fit(X)
    assert np.isfinite(model.memberships_).all()
    np.testing.assert_allclose(model.memberships_.sum(1), 1)
    assert np.all(np.diff(model.log_likelihoods) >= -1e-10)
    with pytest.raises(ValueError, match="integer counts.*plain EM accepts"):
        PLSI(n_clusters=2, tempered=True).fit(X)


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_tem_requires_an_interior_holdout_ratio(ratio):
    with pytest.raises(ValueError, match="0 < heldout_ratio < 1"):
        PLSI(n_clusters=2, tempered=True, heldout_ratio=ratio).fit(
            sp.csr_matrix([[2, 1], [1, 2]])
        )


def test_tem_canonicalizes_private_counts_without_mutating_input():
    X = sp.csr_matrix(
        (
            np.array([2.0, 3.0, 0.0, 4.0, 2.0]),
            np.array([0, 0, 1, 0, 1]),
            np.array([0, 3, 5]),
        ),
        shape=(2, 2),
    )
    saved = (X.data.copy(), X.indices.copy(), X.indptr.copy())
    result = PLSI(
        n_clusters=2, n_init=1, max_iter=3, random_state=4, tempered=True
    ).fit(X)
    np.testing.assert_allclose(result.memberships_.sum(1), 1)
    assert np.isfinite(result.perplexity)
    for actual, original in zip((X.data, X.indices, X.indptr), saved):
        np.testing.assert_array_equal(actual, original)


def test_tem_restores_selected_restart_parameters_history_and_diagnostics(monkeypatch):
    states = []
    splits = []

    def controlled_tem(self, X, train, held_out):
        i = len(states)
        self.P_z = np.array([0.2 + i * 0.1, 0.8 - i * 0.1])
        self.P_d_given_z = np.array([[0.6, 0.4], [0.3, 0.7]])
        self.P_w_given_z = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.log_likelihoods = [-10.0 + i]
        self.heldout_log_likelihood_ = [-4.0, -1.0, -3.0][i]
        self.beta_ = [0.9, 0.81, 0.729][i]
        states.append(
            (self.P_z.copy(), self.P_d_given_z.copy(), self.P_w_given_z.copy())
        )
        splits.append((train, held_out))

    monkeypatch.setattr(PLSI, "_tempered_em", controlled_tem)
    model = PLSI(n_clusters=2, n_init=3, random_state=0, tempered=True).fit(
        sp.csr_matrix([[20, 10], [10, 20]])
    )
    for actual, selected in zip(
        (model.P_z, model.P_d_given_z, model.P_w_given_z), states[1]
    ):
        np.testing.assert_array_equal(actual, selected)
    assert model.log_likelihoods == [-9.0]
    assert model.heldout_log_likelihood_ == -1.0
    assert (
        model.beta_ == 0.81
    )  # Selected run's terminal beta, not a new checkpoint definition.
    np.testing.assert_allclose(model.memberships_.sum(1), 1)
    assert np.isfinite(model.perplexity)
    for split in splits[1:]:
        for current, original in zip(split, splits[0]):
            for array, shared in zip(current, original):
                assert array is shared
