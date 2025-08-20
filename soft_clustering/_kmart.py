import numpy as np
import random
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
from typing import List, Set, Dict, Tuple

class KMART:
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
        self.clusters_: List[Set[int]] = []
        self.prototypes_: List[np.ndarray] = []
        self._unique_words: List[str] = []
        self.cluster_words_: List[Set[str]] = []

    def _preprocess(self, docs: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Transforms documents into a vector representation (bag-of-words).
        Removes stop words and creates a unique vocabulary.

        Args:
            docs (List[str]): A list of text documents.

        Returns:
            Tuple[List[np.ndarray], List[str]]: A tuple containing a list of document
                                                 vectors and the unique vocabulary.
        """
        stop_words = set([
            "the", "a", "an", "and", "or", "but", "in", "on", "with", "for",
            "to", "of", "from", "at", "by", "is", "are", "be", "was", "were",
            "it", "its", "that", "this", "these", "those"
        ])

        word_counts = []
        unique_words = set()

        # First pass: Build word counts for each document and the global vocabulary
        for doc in docs:
            words = doc.lower().split()
            doc_word_counts = defaultdict(int)
            for word in words:
                if word not in stop_words:
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

    def _extract_keywords(self, docs: List[str]) -> List[Set[str]]:
        """
        Extracts representative keywords for each cluster by collecting all
        words from the documents within each final cluster.
        
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
                    word_counts[word] += 1
            sorted_words = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)
            cluster_keywords.append(set(sorted_words[:10]))
        return cluster_keywords

    def fit_predict(self, docs: List[str]) -> csr_matrix:
        """
        Runs the KMART clustering algorithm on a collection of documents.

        Args:
            docs (List[str]): A list of text documents.

        Returns:
            csr_matrix: A sparse matrix of shape (num_docs, num_clusters) indicating
                        document membership in the final clusters.
        """
        doc_vectors, self._unique_words = self._preprocess(docs)
        
        for i, doc_vector in enumerate(doc_vectors):
            
            # Find all prototypes that pass the vigilance test
            passed_tests = []
            for j, prototype in enumerate(self.prototypes_):
                fuzzy_and_result = self._fuzzy_and(doc_vector, prototype)
                # Vigilance Test: ||I & P|| / ||I|| >= rho
                # L1 norm is used for the vectors
                vigilance_score = np.sum(fuzzy_and_result) / (np.sum(doc_vector) + 1e-9) # Add a small epsilon to avoid division by zero
                
                if vigilance_score >= self.vigilance_param:
                    passed_tests.append(j)

            # If no prototypes pass, create a new cluster (unsupervised learning)
            if not passed_tests:
                # Initialize a new prototype with the current document vector
                self.prototypes_.append(doc_vector)
                # Create a new cluster and add the document to it
                self.clusters_.append({i})
            else:
                # If one or more prototypes pass, update all of them
                for cluster_idx in passed_tests:
                    prototype = self.prototypes_[cluster_idx]
                    
                    # Update rule: P_new = lambda * (I & P_old) + (1 - lambda) * P_old
                    updated_prototype = self.learning_rate * self._fuzzy_and(doc_vector, prototype) + (1 - self.learning_rate) * prototype
                    self.prototypes_[cluster_idx] = updated_prototype
                    
                    # Add the document to the corresponding cluster
                    self.clusters_[cluster_idx].add(i)
        
        # Post-processing: Generate the output membership matrix and keywords
        self.cluster_words_ = self._extract_keywords(docs)

        num_docs = len(docs)
        num_clusters = len(self.clusters_)
        memberships = lil_matrix((num_docs, num_clusters), dtype=np.int8)
        
        for cluster_idx, doc_set in enumerate(self.clusters_):
            for doc_idx in doc_set:
                memberships[doc_idx, cluster_idx] = 1
                
        return memberships.tocsr()
