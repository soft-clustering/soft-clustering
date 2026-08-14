# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""KMART — a modified Fuzzy ART for soft document clustering.

Implementation note (optimization study)
----------------------------------------
Vectorised implementation of the same algorithm. The vigilance test, the
prototype update rule, the order in which documents are presented and the
order in which prototypes are created and updated are unchanged. The reference
implementation is preserved at
``optimization/original/scpp_original/_kmart.py``.

Profiling attributed the runtime to 28,634 ``np.sum`` calls inside
``_fuzzy_and`` — one scalar reduction per (document, prototype) pair. Because
``_fuzzy_and`` is just ``np.minimum``, the whole vigilance test over the
current category set is a single broadcast:

    scores = sum(minimum(I, P), axis=1) / (sum(I) + eps)

with ``P`` the block of existing prototypes. Prototypes are therefore held in
one contiguous ``(capacity, vocab)`` buffer, doubled on demand, instead of a
Python list of separate vectors, and the passing categories are updated with a
single fancy-indexed assignment rather than one at a time.

The membership matrix is assembled in COO form; the reference assigned into a
``lil_matrix`` element by element, which costs a Python-level index operation
per (document, cluster) pair.

``prototypes_`` is still published as the documented list of per-cluster
vectors, and ``_fuzzy_and`` is retained as part of the class's interface.
"""

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ._base import BaseSoftClusterer


class KMART(BaseSoftClusterer):
    # ART vigilance yields independent per-category activations
    _partition_constrained = False
    # Discovers the cluster count; see BaseSoftClusterer._determines_k.
    _determines_k = True
    """
    Implements a modified Fuzzy Adaptive Resonance Theory (Fuzzy ART) algorithm
    for soft document clustering.

    This algorithm, named KMART, adapts Fuzzy ART to enable a document to
    be in multiple clusters, making it suitable for multi-topic documents.
    It removes the iterative search for a "winning" category, leading to a
    more efficient clustering process.

    Attributes:
        vigilance_param (float): The vigilance parameter (rho), between 0 and 1.
                                 A higher value leads to more specific (tighter) clusters.
        learning_rate (float): The learning rate (lambda), between 0 and 1.
                               If set to 1.0 (fast learning), the prototype is
                               immediately updated to match the input.
        clusters_ (List[Set[int]]): A list of sets, where each set contains the
                                     document indices belonging to a cluster.
        prototypes_ (List[np.ndarray]): A list of the final prototype vectors for each cluster.
        cluster_words_ (List[Set[str]]): The representative words for each final cluster.
    """

    def __init__(self, vigilance_param: float = 0.5, learning_rate: float = 1.0):
        """
        Initializes the KMART algorithm's parameters.

        Args:
            vigilance_param (float): The vigilance parameter (rho) for the vigilance test.
            learning_rate (float): The learning rate (lambda) for updating prototypes.
        """
        self.vigilance_param = vigilance_param
        self.learning_rate = learning_rate
        self.clusters_: list[set[int]] = []
        self.prototypes_: list[np.ndarray] = []
        self._unique_words: list[str] = []
        self.cluster_words_: list[set[str]] = []

        # Stop words are defined as a class attribute for reusability.
        self._stop_words = set(
            [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "with",
                "for",
                "to",
                "of",
                "from",
                "at",
                "by",
                "is",
                "are",
                "be",
                "was",
                "were",
                "it",
                "its",
                "that",
                "this",
                "these",
                "those",
            ]
        )

    def _preprocess(self, docs: list[str]) -> tuple[list[np.ndarray], list[str]]:
        """
        Transforms documents into a vector representation (bag-of-words).
        Removes stop words and creates a unique vocabulary.

        Args:
            docs (List[str]): A list of text documents.

        Returns:
            Tuple[List[np.ndarray], List[str]]: A tuple containing a list of document
                                                 vectors and the unique vocabulary.
        """
        word_counts = []
        unique_words = set()

        # First pass: Build word counts for each document and the global vocabulary
        for doc in docs:
            words = doc.lower().split()
            doc_word_counts = defaultdict(int)
            for word in words:
                # Use the class's stop word set for filtering
                if word not in self._stop_words:
                    doc_word_counts[word] += 1
                    unique_words.add(word)
            word_counts.append(doc_word_counts)

        # Create a sorted list of unique words for consistent vector indexing
        self._unique_words = sorted(list(unique_words))
        word_to_idx = {word: i for i, word in enumerate(self._unique_words)}

        # Second pass: Create document frequency vectors
        doc_vectors = []
        vocab_size = len(self._unique_words)
        for counts in word_counts:
            vector = np.zeros(vocab_size)
            for word, count in counts.items():
                if word in word_to_idx:
                    vector[word_to_idx[word]] = count
            doc_vectors.append(vector)

        return doc_vectors, self._unique_words

    def _fuzzy_and(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Calculates the Fuzzy AND (min) between two vectors.
        This is used to find the intersection of features.

        Args:
            vec1 (np.ndarray): The first vector.
            vec2 (np.ndarray): The second vector.

        Returns:
            np.ndarray: The resulting vector from the Fuzzy AND operation.
        """
        return np.minimum(vec1, vec2)

    def _extract_keywords(self, docs: list[str]) -> list[set[str]]:
        """
        Extracts representative keywords for each cluster by collecting all
        words from the documents within each final cluster, filtering out stop words.

        Args:
            docs (List[str]): The original list of text documents.

        Returns:
            List[Set[str]]: A list of sets, where each set contains the keywords
                            for a corresponding cluster.
        """
        cluster_keywords = []
        for doc_set in self.clusters_:
            word_counts = defaultdict(int)
            for doc_idx in doc_set:
                words = docs[doc_idx].lower().split()
                for word in words:
                    # Filter out stop words here to ensure the final output is clean
                    if word not in self._stop_words:
                        word_counts[word] += 1
            sorted_words = sorted(
                word_counts.keys(), key=lambda w: word_counts[w], reverse=True
            )
            cluster_keywords.append(set(sorted_words[:10]))
        return cluster_keywords

    def fit_predict(self, docs: list[str]) -> csr_matrix:
        """
        Runs the KMART clustering algorithm on a collection of documents.

        Args:
            docs (List[str]): A list of text documents.

        Returns:
            csr_matrix: A sparse matrix of shape (num_docs, num_clusters) indicating
                        document membership in the final clusters.
        """
        doc_vectors, self._unique_words = self._preprocess(docs)

        vocab_size = len(self._unique_words)

        # Prototypes are held in one contiguous (capacity, vocab) buffer so the
        # vigilance test is a single broadcast rather than a Python loop over
        # categories. Capacity doubles on demand, so growing the set of
        # prototypes stays amortised O(1) per new cluster.
        capacity = 8
        prototypes = np.empty((capacity, vocab_size), dtype=np.float64)
        n_prototypes = 0

        for i, doc_vector in enumerate(doc_vectors):

            # Vigilance test against every existing prototype at once:
            #   ||I & P_j||_1 / ||I||_1 >= rho
            # np.minimum is the Fuzzy AND; broadcasting the document against
            # the prototype block gives all scores in one reduction.
            denominator = np.sum(doc_vector) + 1e-9  # avoid division by zero
            active = prototypes[:n_prototypes]
            scores = np.sum(np.minimum(doc_vector, active), axis=1) / denominator
            passed_tests = np.flatnonzero(scores >= self.vigilance_param)

            # If no prototypes pass, create a new cluster (unsupervised learning)
            if passed_tests.size == 0:
                if n_prototypes == capacity:
                    capacity *= 2
                    grown = np.empty((capacity, vocab_size), dtype=np.float64)
                    grown[:n_prototypes] = prototypes[:n_prototypes]
                    prototypes = grown
                # Initialize a new prototype with the current document vector
                prototypes[n_prototypes] = doc_vector
                n_prototypes += 1
                # Create a new cluster and add the document to it
                self.clusters_.append({i})
            else:
                # If one or more prototypes pass, update all of them.
                # Update rule: P_new = lambda * (I & P_old) + (1 - lambda) * P_old
                selected = prototypes[passed_tests]
                prototypes[passed_tests] = (
                    self.learning_rate * np.minimum(doc_vector, selected)
                    + (1 - self.learning_rate) * selected
                )

                # Add the document to the corresponding clusters
                for cluster_idx in passed_tests:
                    self.clusters_[cluster_idx].add(i)

        # Publish the prototypes in the documented form: one array per cluster.
        self.prototypes_ = [prototypes[j].copy() for j in range(n_prototypes)]

        # Post-processing: Generate the output membership matrix and keywords
        self.cluster_words_ = self._extract_keywords(docs)

        num_docs = len(docs)
        num_clusters = len(self.clusters_)

        # Built directly in COO form: assigning into a lil_matrix cost one
        # Python-level index operation per (document, cluster) pair.
        rows = [doc_idx for doc_set in self.clusters_ for doc_idx in doc_set]
        cols = [
            cluster_idx
            for cluster_idx, doc_set in enumerate(self.clusters_)
            for _ in doc_set
        ]
        memberships = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(num_docs, num_clusters),
            dtype=np.int8,
        )

        return memberships.tocsr()
