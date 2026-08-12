import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import psi
from sklearn.feature_extraction.text import CountVectorizer
from typeguard import typechecked

from ._base import BaseSoftClusterer


class LDA(BaseSoftClusterer):
    @typechecked
    def __init__(
        self,
        n_topics: int = 10,
        alpha: float | None = None,
        beta: float = 0.01,
        max_iter: int = 100,
        var_max_iter: int = 20,
        tol: float = 1e-4,
        n_init: int = 5,
        random_state: int | None = None,
    ):
        """
        Latent Dirichlet Allocation (LDA) with variational EM inference.

        Parameters
        ----------
        n_topics : int
            Number of latent topics to infer.
        alpha : float or None
            Dirichlet prior for document-topic distribution.
        beta : float
            Dirichlet prior for topic-word distribution.
        max_iter : int
            Number of outer EM iterations.
        var_max_iter : int
            Number of variational inference steps per document (inner loop).
        tol : float
            Convergence tolerance on mean change in gamma.
        n_init : int
            Number of random restarts; the run with the highest corpus
            log-likelihood is kept. Variational EM for LDA has many local
            optima, and a single run collapses two topics into one on roughly
            one initialisation in five, even on a corpus with three cleanly
            separated topics.
        random_state : int or None
            Seed for the variational initialisation. Previously the
            initialisation drew from NumPy's global generator, so a fit could
            not be reproduced without seeding that generator externally.
        """
        self.n_topics = n_topics
        self.alpha = alpha if alpha is not None else 50.0 / n_topics
        self.beta = beta
        self.max_iter = max_iter
        self.var_max_iter = var_max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def _initialize(self, X, seed=None):
        """
        Initialize model parameters: gamma (D x K), lambda (K x V), and alpha vector.
        """
        D, V = X.shape
        rng = np.random.default_rng(seed)

        # Gamma: document-topic Dirichlet parameters
        self.gamma = rng.gamma(100.0, 1.0 / 100.0, (D, self.n_topics))

        # Lambda: topic-word Dirichlet parameters
        self.lambda_ = rng.gamma(100.0, 1.0 / 100.0, (self.n_topics, V)) + self.beta

        # Alpha vector: symmetric prior over topics
        self.alpha_vec = np.full(self.n_topics, self.alpha)

    def fit(self, X, vocabulary=None):
        """
        Run variational EM algorithm to fit LDA model to corpus.

        Parameters
        ----------
        X : list of str or csr_matrix
            Input documents as raw strings or term-document matrix.
        vocabulary : list[str], optional
            Vocabulary to fix during CountVectorizer use.

        Returns
        -------
        self : LDA
            Trained model instance.
        """
        # Convert text corpus to bag-of-words matrix if needed
        if not isinstance(X, (csr_matrix, np.ndarray)):
            vectorizer = CountVectorizer(vocabulary=vocabulary)
            X = vectorizer.fit_transform(X)
            self.vocab_ = vectorizer.get_feature_names_out()
        else:
            X = csr_matrix(X)
            if vocabulary is not None:
                self.vocab_ = np.array(vocabulary)
            else:
                self.vocab_ = np.array([f"w{i}" for i in range(X.shape[1])])

        X_csr = X.tocsr()

        # Variational EM is run from n_init random initialisations and the run
        # with the highest corpus log-likelihood is kept. A single run merges
        # two topics into one on roughly one initialisation in five, even on a
        # corpus of three cleanly separated topics.
        base_seed = self.random_state if self.random_state is not None else 0
        best = (-np.inf, None, None)
        for restart in range(self.n_init):
            self._run_em(X_csr, seed=base_seed + restart)
            score = self._corpus_log_likelihood(X_csr)
            if score > best[0]:
                best = (score, self.gamma.copy(), self.lambda_.copy())

        self.log_likelihood_, self.gamma, self.lambda_ = best

        # The variational Dirichlet parameters gamma, normalised, are the
        # document-topic distributions -- a soft partition of the documents
        # over topics. Publishing them is what lets LDA be evaluated and
        # conformance-checked alongside every other estimator; the topic-word
        # side of the model stays available through get_topic_word_dist().
        self.doc_topic_ = self.gamma / self.gamma.sum(axis=1, keepdims=True)
        self.memberships_ = self.doc_topic_
        return self

    def _corpus_log_likelihood(self, X_csr) -> float:
        """``sum_d sum_w n_dw log(sum_k theta_dk beta_kw)``.

        The standard predictive likelihood used to compare LDA fits. Higher is
        better; it is what selects between restarts.
        """
        theta = self.gamma / self.gamma.sum(axis=1, keepdims=True)
        beta = self.lambda_ / self.lambda_.sum(axis=1, keepdims=True)
        total = 0.0
        for d in range(X_csr.shape[0]):
            ids, counts = X_csr[d].indices, X_csr[d].data
            if ids.size == 0:
                continue
            word_prob = theta[d] @ beta[:, ids]
            total += float(np.dot(counts, np.log(np.maximum(word_prob, 1e-300))))
        return total

    def _run_em(self, X_csr, seed=None) -> None:
        """One variational EM run from a fresh initialisation."""
        self._initialize(X_csr, seed=seed)

        # Begin EM iterations
        for em_iter in range(self.max_iter):
            lambda_sum = self.lambda_.sum(axis=1)  # For digamma normalization
            gamma_old = self.gamma.copy()  # Save gamma for convergence check
            accum_lambda = np.zeros_like(
                self.lambda_
            )  # Accumulator for expected word-topic counts

            # ----------- E-Step -----------
            for d in range(X_csr.shape[0]):
                ids = X_csr[d].indices  # Word indices in doc d
                counts = X_csr[d].data  # Word counts in doc d

                # Initialize gamma_d and phi_dn uniformly
                gamma_d = self.alpha_vec + np.sum(counts) / self.n_topics
                phi_dn = np.full((len(ids), self.n_topics), 1.0 / self.n_topics)

                # Variational updates for document d
                for _ in range(self.var_max_iter):
                    dig_gamma = psi(gamma_d)

                    # Update phi: topic responsibilities for each word
                    log_phi = dig_gamma + psi(self.lambda_[:, ids]).T - psi(lambda_sum)
                    phi_dn = np.exp(log_phi)
                    phi_dn /= phi_dn.sum(axis=1, keepdims=True)

                    # Update gamma: topic distribution for document d
                    gamma_d = self.alpha_vec + np.dot(counts, phi_dn)

                # Store updated gamma
                self.gamma[d] = gamma_d

                # Accumulate sufficient statistics for lambda update
                accum_lambda[:, ids] += phi_dn.T * counts

            # ----------- M-Step -----------
            self.lambda_ = self.beta + accum_lambda

            # ----------- Convergence Test -----------
            mean_change = np.mean(np.abs(self.gamma - gamma_old))
            if mean_change < self.tol:
                break

    def get_doc_topic_dist(self):
        """
        Return the normalized document-topic distributions.

        Returns
        -------
        doc_topic : ndarray of shape (n_documents, n_topics)
            Each row is a probability distribution over topics for a document.
            Identical to ``memberships_``.
        """
        self._check_fitted()
        return self.doc_topic_

    def get_topic_word_dist(self):
        """
        Return the normalized topic-word distributions.

        Returns
        -------
        topic_word : ndarray of shape (n_topics, V)
            Each row is a probability distribution over vocabulary for a topic.
        """
        return self.lambda_ / self.lambda_.sum(axis=1, keepdims=True)

    def print_top_words(self, n_top_words=10):
        """
        Print top words per topic based on learned topic-word distributions.

        Parameters
        ----------
        n_top_words : int
            Number of top words to display per topic.
        """
        topic_word = self.get_topic_word_dist()
        for k in range(self.n_topics):
            top_indices = topic_word[k].argsort()[::-1][:n_top_words]
            words = [self.vocab_[i] for i in top_indices]
            print(f"Topic {k}: {' '.join(words)}")
