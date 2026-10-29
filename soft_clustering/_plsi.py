# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_random_state


def _split_tokens(X, rng, heldout_ratio):
    """Split canonical CSR integer counts, retaining training support per document."""
    if not 0 < heldout_ratio < 1:
        raise ValueError("PLSI: TEM requires 0 < heldout_ratio < 1.")
    if np.any(X.data != np.floor(X.data)):
        raise ValueError(
            "PLSI: token-level TEM holdout requires integer counts; "
            "plain EM accepts fractional weighted inputs."
        )
    if np.any(X.data >= 2**63):
        raise ValueError("PLSI: TEM token counts must fit in signed 64-bit integers.")
    counts = X.data.astype(np.int64)
    heldout = rng.binomial(counts, heldout_ratio)
    train = counts - heldout
    for start, end in zip(X.indptr[:-1], X.indptr[1:]):
        if train[start:end].sum() == 0:
            positions = np.arange(start, end)
            chosen = rng.choice(
                positions, p=heldout[start:end] / heldout[start:end].sum()
            )
            train[chosen] += 1
            heldout[chosen] -= 1
    if heldout.sum() == 0:
        totals = np.add.reduceat(train, X.indptr[:-1])
        eligible = np.flatnonzero(
            np.repeat(totals > 1, np.diff(X.indptr)) & (train > 0)
        )
        if eligible.size == 0:
            raise ValueError(
                "PLSI: a nonempty TEM holdout needs at least one document with "
                "two tokens while retaining training tokens in every document."
            )
        chosen = rng.choice(eligible, p=train[eligible] / train[eligible].sum())
        train[chosen] -= 1
        heldout[chosen] += 1
    coo = X.tocoo()
    return tuple(
        (coo.row[values > 0], coo.col[values > 0], values[values > 0])
        for values in (train, heldout)
    )


class PLSI(BaseSoftClusterer):
    r"""Probabilistic latent semantic indexing (aspect model) fitted by EM.

    Models a document-word count matrix as :math:`P(d, w) = \sum_z P(z)
    P(d \mid z) P(w \mid z)` and maximises the likelihood by EM (Hofmann,
    1999). ``memberships_`` holds :math:`P(z \mid d) \propto P(d \mid z)
    P(z)` in document-by-topic orientation; legacy ``P_z_given_d`` remains
    topic-by-document. The training log-likelihood (``log_likelihoods``) is non-decreasing
    under plain EM. Tempered EM, with Hofmann's annealing schedule and a
    held-out set, is available through ``tempered=True``; it targets
    generalisation to unseen documents and needs a large corpus. Parameters
    are documented under ``__init__`` below.

    References
    ----------
    T. Hofmann. Probabilistic latent semantic analysis. In *Proc. 15th
    Conference on Uncertainty in Artificial Intelligence (UAI)*, pp. 289--296,
    1999.
    """

    # Fitted attributes, set by ``fit`` (not before).
    P_z: np.ndarray
    P_d_given_z: np.ndarray
    P_w_given_z: np.ndarray
    P_z_given_d: np.ndarray
    log_likelihoods: list[float]

    @typechecked
    def __init__(
        self,
        n_clusters: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        tempered: bool = False,
        beta_start: float = 1.0,
        beta_step: float = 0.9,
        heldout_ratio: float = 0.1,
        n_init: int = 3,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_clusters : int, default=10
            Number of latent topics (``n_topics`` is accepted as an alias).
        max_iter : int, default=100
            Maximum number of EM iterations.
        tol : float, default=1e-4
            Convergence threshold for log-likelihood.
        tempered : bool, default=False
            Use tempered EM (Hofmann, 1999) with a held-out set for early
            stopping and annealing, instead of plain EM. TEM is designed to
            improve generalisation to *unseen* documents and needs a large
            corpus: with a small held-out set its early stopping is driven by
            noise and can halt within a few iterations of a random start. For
            clustering the documents given to ``fit`` plain EM (the default
            since 0.2.0) is the appropriate choice. With ``tempered=True``
            the held-out tokens are not used for training.
        beta_start : float, default=1.0
            Initial value of inverse temperature beta.
        beta_step : float, default=0.9
            Multiplicative step to reduce beta at each iteration.
        heldout_ratio : float, default=0.1
            Approximate fraction of integer word tokens held out by TEM;
            must be strictly between zero and one when tempered=True. Every
            document retains a training token, and validation retains at least
            one token globally. Plain EM also accepts fractional weights.
        n_init : int, default=3
            Number of random restarts. The run with the highest training
            log-likelihood (plain EM) or held-out log-likelihood (tempered
            EM) is kept.
        random_state : int or None, default=None
            Seed for random number generator.
        """
        self.n_clusters = n_clusters
        self.n_topics = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.tempered = tempered
        self.beta_start = beta_start
        self.beta_step = beta_step
        self.heldout_ratio = heldout_ratio
        self.n_init = n_init
        self.random_state = random_state

        self.log_likelihoods = []
        self.perplexity = None
        self.vocab = None

    def _initialize(self, n_docs, n_words):
        """Randomly initialize model parameters."""
        self.P_z = np.ones(self.n_topics) / self.n_topics
        self.P_w_given_z = self._rng.dirichlet(np.ones(n_words), self.n_topics)
        self.P_d_given_z = self._rng.dirichlet(np.ones(n_docs), self.n_topics)

    def _e_step(self, doc_idx, word_idx, beta):
        """
        Perform the E-step of the EM algorithm.

        Parameters
        ----------
        doc_idx : ndarray
            Document indices of non-zero entries in the term-document matrix.
        word_idx : ndarray
            Word indices of non-zero entries.
        beta : float
            Inverse temperature parameter for tempered EM.

        Returns
        -------
        ndarray
            Posterior topic probabilities P(z | d, w).
        """
        # P(z | d, w) proportional to P(z) * [P(d | z) P(w | z)]^beta
        # (Hofmann, 1999, tempered E-step), evaluated in log space so that
        # large corpora, whose P(d|z) P(w|z) underflow, cannot produce 0/0.
        tiny = np.finfo(np.float64).tiny
        log_w = np.log(np.maximum(self.P_z, tiny))[None, :] + beta * (
            np.log(np.maximum(self.P_d_given_z[:, doc_idx].T, tiny))
            + np.log(np.maximum(self.P_w_given_z[:, word_idx].T, tiny))
        )
        log_w -= log_w.max(axis=1, keepdims=True)
        weight = np.exp(log_w)
        weight /= weight.sum(axis=1, keepdims=True)
        return weight

    def _m_step(self, X, doc_idx, word_idx, count, p_z_dw):
        """
        Perform the M-step of the EM algorithm.

        Parameters
        ----------
        X : csr_matrix
            Sparse term-document matrix.
        doc_idx : ndarray
            Document indices.
        word_idx : ndarray
            Word indices.
        count : ndarray
            Word counts.
        p_z_dw : ndarray
            Posterior topic probabilities from E-step.
        """
        n_docs = X.shape[0]
        n_words = X.shape[1]

        # Update P(w | z)
        self.P_w_given_z = np.zeros((self.n_topics, n_words))
        for z in range(self.n_topics):
            np.add.at(self.P_w_given_z[z], word_idx, count * p_z_dw[:, z])
        self.P_w_given_z /= self.P_w_given_z.sum(axis=1, keepdims=True)

        # Update P(d | z)
        self.P_d_given_z = np.zeros((self.n_topics, n_docs))
        for z in range(self.n_topics):
            np.add.at(self.P_d_given_z[z], doc_idx, count * p_z_dw[:, z])
        self.P_d_given_z /= self.P_d_given_z.sum(axis=1, keepdims=True)

        # Update P(z)
        self.P_z = (count[:, None] * p_z_dw).sum(axis=0)
        self.P_z /= self.P_z.sum()

    def _log_likelihood(self, X, doc_idx, word_idx, count):
        """
        Compute the log-likelihood of the model.

        Parameters
        ----------
        X : csr_matrix
            Sparse term-document matrix.
        doc_idx : ndarray
            Document indices.
        word_idx : ndarray
            Word indices.
        count : ndarray
            Word counts.

        Returns
        -------
        float
            Log-likelihood value.
        """
        prob = np.zeros(len(doc_idx))
        for z in range(self.n_topics):
            prob += (
                self.P_z[z]
                * self.P_d_given_z[z, doc_idx]
                * self.P_w_given_z[z, word_idx]
            )
        return np.sum(count * np.log(prob + 1e-12))

    def _perplexity(self, X, doc_idx, word_idx, count):
        """
        Compute the perplexity of the model.

        Parameters
        ----------
        X : csr_matrix
            Sparse term-document matrix.
        doc_idx : ndarray
            Document indices.
        word_idx : ndarray
            Word indices.
        count : ndarray
            Word counts.

        Returns
        -------
        float
            Perplexity value.
        """
        ll = self._log_likelihood(X, doc_idx, word_idx, count)
        total = count.sum()
        return np.exp(-ll / total)

    def _tempered_em(self, X, train, held_out):
        """Tempered EM with Hofmann's (1999) annealing schedule.

        1. Plain EM (``beta = beta_start``) until the held-out likelihood stops
           improving (early stopping).
        2. Lower the inverse temperature, ``beta <- beta_step * beta``, and keep
           iterating while the held-out likelihood improves.
        3. Stop when lowering ``beta`` no longer improves it, and keep the
           parameters with the best held-out likelihood.

        Earlier releases lowered ``beta`` at every iteration from the start
        and stopped at the first dip of the held-out likelihood, which on a
        random initialisation usually happened after two iterations.
        """
        doc_t, word_t, count_t = train
        doc_v, word_v, count_v = held_out

        def snapshot():
            return (self.P_z.copy(), self.P_d_given_z.copy(), self.P_w_given_z.copy())

        def improves(value, reference):
            if not np.isfinite(reference):
                return bool(np.isfinite(value))
            return value > reference + self.tol * max(abs(reference), 1.0)

        beta = self.beta_start
        best_val = -np.inf
        best = snapshot()
        iterations = 0
        while iterations < self.max_iter:
            improved_at_this_beta = False
            while iterations < self.max_iter:
                p_z_dw = self._e_step(doc_t, word_t, beta)
                self._m_step(X, doc_t, word_t, count_t, p_z_dw)
                iterations += 1
                self.log_likelihoods.append(
                    self._log_likelihood(X, doc_t, word_t, count_t)
                )
                val = self._log_likelihood(X, doc_v, word_v, count_v)
                if improves(val, best_val):
                    best_val, best = val, snapshot()
                    improved_at_this_beta = True
                else:
                    break
            if not improved_at_this_beta:
                break
            beta *= self.beta_step
            # Restart the next temperature from the best parameters so far.
            self.P_z, self.P_d_given_z, self.P_w_given_z = (a.copy() for a in best)

        self.P_z, self.P_d_given_z, self.P_w_given_z = best
        self.beta_ = beta
        self.heldout_log_likelihood_ = best_val

    def fit_predict(self, data: list[str] | csr_matrix):
        """
        Fit the PLSI model to a corpus.

        Parameters
        ----------
        data : list of str or csr_matrix
            Raw text corpus or a sparse term-document matrix. Counts must be
            finite and nonnegative. Empty documents are rejected: their topic
            proportions cannot be inferred without observed tokens.
        """
        # Every fit starts from the constructor's seed and an empty history.
        # Earlier releases stored a RandomState in ``random_state`` itself,
        # so successive fits of one instance drew different initialisations,
        # and appended to the log-likelihoods of every previous fit.
        self.n_topics = self.n_clusters
        self._rng = check_random_state(self.random_state)
        self.log_likelihoods = []
        if isinstance(data, list):
            vectorizer = CountVectorizer()
            try:
                X = vectorizer.fit_transform(data)
            except ValueError as exc:
                if "empty vocabulary" in str(exc):
                    raise ValueError(
                        "PLSI: the corpus has no observed word tokens."
                    ) from exc
                raise
            self.vocab = vectorizer.get_feature_names_out()
        else:
            X = data

        n_docs, n_words = X.shape
        X = csr_matrix(X, copy=True)
        if not np.isfinite(X.data).all() or np.any(X.data < 0):
            raise ValueError("PLSI: counts must be finite and nonnegative.")
        X.sum_duplicates()
        X.eliminate_zeros()
        empty = np.flatnonzero(np.asarray(X.sum(axis=1)).ravel() == 0)
        if n_docs == 0 or empty.size:
            raise ValueError(
                f"PLSI: empty documents at indices {empty.tolist()}; cannot infer "
                "topic proportions without observed tokens."
            )
        coo = X.tocoo()
        doc_idx, word_idx, count = coo.row, coo.col, coo.data

        # Split held-out if TEM
        if self.tempered:
            # Generator supports int64 binomial counts even where RandomState
            # uses a 32-bit C long (Windows with older NumPy releases).
            split_rng = np.random.default_rng(self._rng.randint(0, 2**31 - 1))
            (doc_idx_train, word_idx_train, count_train), (
                doc_idx_val,
                word_idx_val,
                count_val,
            ) = _split_tokens(X, split_rng, self.heldout_ratio)
        else:
            doc_idx_train, word_idx_train, count_train = doc_idx, word_idx, count
            doc_idx_val, word_idx_val, count_val = None, None, None

        if self.n_init < 1:
            raise ValueError(f"PLSI: n_init must be >= 1; got {self.n_init}.")
        best_score, best_state = -np.inf, None
        best_diagnostics = None
        for _ in range(self.n_init):
            self.log_likelihoods = []
            self._initialize(n_docs, n_words)
            if not self.tempered:
                for i in range(self.max_iter):
                    p_z_dw = self._e_step(doc_idx_train, word_idx_train, 1.0)
                    self._m_step(X, doc_idx_train, word_idx_train, count_train, p_z_dw)
                    ll = self._log_likelihood(
                        X, doc_idx_train, word_idx_train, count_train
                    )
                    self.log_likelihoods.append(ll)
                    if i > 0 and abs(ll - self.log_likelihoods[-2]) < self.tol:
                        break
                run_score = self.log_likelihoods[-1]
            else:
                self._tempered_em(
                    X,
                    (doc_idx_train, word_idx_train, count_train),
                    (doc_idx_val, word_idx_val, count_val),
                )
                run_score = self.heldout_log_likelihood_
            if best_state is None or run_score > best_score:
                best_score = run_score
                best_state = (
                    self.P_z.copy(),
                    self.P_d_given_z.copy(),
                    self.P_w_given_z.copy(),
                    list(self.log_likelihoods),
                )
                if self.tempered:
                    best_diagnostics = (self.beta_, self.heldout_log_likelihood_)
        assert best_state is not None  # n_init >= 1 runs were made
        self.P_z, self.P_d_given_z, self.P_w_given_z, self.log_likelihoods = best_state
        if best_diagnostics is not None:
            self.beta_, self.heldout_log_likelihood_ = best_diagnostics

        self.perplexity = self._perplexity(X, doc_idx, word_idx, count)
        self.P_z_given_d = (self.P_d_given_z.T * self.P_z).T
        self.P_z_given_d /= self.P_z_given_d.sum(axis=0, keepdims=True)
        self.memberships_ = self.P_z_given_d.T

        return self

    def get_P_w_given_d(self):
        """
        Compute the smoothed word distributions per document.

        Returns
        -------
        ndarray
            Matrix of P(w | d) values.
        """
        return self.P_z_given_d.T @ self.P_w_given_z
