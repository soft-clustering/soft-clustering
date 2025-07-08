import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import CountVectorizer
from typing import Union, Optional, List
from typeguard import typechecked


class PLSI:
    @typechecked
    def __init__(
        self,
        n_topics: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        tempered: bool = True,
        beta_start: float = 1.0,
        beta_step: float = 0.9,
        heldout_ratio: float = 0.1,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_topics : int, default=10
            Number of latent topics.
        max_iter : int, default=100
            Maximum number of EM iterations.
        tol : float, default=1e-4
            Convergence threshold for log-likelihood.
        tempered : bool, default=True
            Whether to use Tempered EM or standard EM.
        beta_start : float, default=1.0
            Initial value of inverse temperature beta.
        beta_step : float, default=0.9
            Multiplicative step to reduce beta at each iteration.
        heldout_ratio : float, default=0.1
            Fraction of word tokens used as held-out validation data.
        random_state : int or None, default=None
            Seed for random number generator.
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.tempered = tempered
        self.beta_start = beta_start
        self.beta_step = beta_step
        self.heldout_ratio = heldout_ratio
        self.random_state = np.random.RandomState(random_state)

        # Outputs
        self.P_w_given_z = None
        self.P_z_given_d = None
        self.log_likelihoods = []
        self.perplexity = None
        self.vocab = None

    def _initialize(self, n_docs, n_words):
        """Randomly initialize model parameters."""
        self.P_z = np.ones(self.n_topics) / self.n_topics
        self.P_w_given_z = self.random_state.dirichlet(np.ones(n_words), self.n_topics)
        self.P_d_given_z = self.random_state.dirichlet(np.ones(n_docs), self.n_topics)

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
        # P(z | d, w) proportional to P(z) * P(d | z)^beta * P(w | z)^beta
        weight = np.zeros((len(doc_idx), self.n_topics))
        for z in range(self.n_topics):
            weight[:, z] = (
                self.P_z[z]
                * np.power(self.P_d_given_z[z, doc_idx], beta)
                * np.power(self.P_w_given_z[z, word_idx], beta)
            )
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

    def fit(self, data: Union[List[str], csr_matrix]):
        """
        Fit the PLSI model to a corpus.

        Parameters
        ----------
        data : list of str or csr_matrix
            Raw text corpus or a sparse term-document matrix.
        """
        if isinstance(data, list):
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(data)
            self.vocab = vectorizer.get_feature_names_out()
        else:
            X = data

        n_docs, n_words = X.shape
        X = csr_matrix(X) if not issparse(X) else X
        X = X.tocoo()
        doc_idx = X.row
        word_idx = X.col
        count = X.data

        # Split held-out if TEM
        if self.tempered:
            mask = self.random_state.rand(len(count)) >= self.heldout_ratio
            train_mask = mask
            heldout_mask = ~mask

            doc_idx_train, word_idx_train, count_train = (
                doc_idx[train_mask],
                word_idx[train_mask],
                count[train_mask],
            )
            doc_idx_val, word_idx_val, count_val = (
                doc_idx[heldout_mask],
                word_idx[heldout_mask],
                count[heldout_mask],
            )
        else:
            doc_idx_train, word_idx_train, count_train = doc_idx, word_idx, count
            doc_idx_val, word_idx_val, count_val = None, None, None

        self._initialize(n_docs, n_words)

        beta = self.beta_start
        best_ll = -np.inf
        for i in range(self.max_iter):
            p_z_dw = self._e_step(doc_idx_train, word_idx_train, beta)
            self._m_step(X, doc_idx_train, word_idx_train, count_train, p_z_dw)

            ll = self._log_likelihood(X, doc_idx_train, word_idx_train, count_train)
            self.log_likelihoods.append(ll)

            if self.tempered:
                val_ll = self._log_likelihood(X, doc_idx_val, word_idx_val, count_val)
                if val_ll < best_ll:
                    break  # early stopping
                best_ll = val_ll
                beta *= self.beta_step

            if (
                i > 0
                and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol
            ):
                break

        self.perplexity = self._perplexity(X, doc_idx, word_idx, count)
        self.P_z_given_d = (self.P_d_given_z.T * self.P_z).T
        self.P_z_given_d /= self.P_z_given_d.sum(axis=0, keepdims=True)

    def get_P_w_given_d(self):
        """
        Compute the smoothed word distributions per document.

        Returns
        -------
        ndarray
            Matrix of P(w | d) values.
        """
        return self.P_z_given_d.T @ self.P_w_given_z
