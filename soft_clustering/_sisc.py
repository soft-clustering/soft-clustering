import numpy as np
import random
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
from typing import List, Set, Dict, Tuple

class SISC:
    """
    Implements the Similarity-based Soft Clustering (SISC) algorithm.

    This algorithm performs soft clustering on documents, meaning a document
    can belong to multiple clusters. It is based on a given similarity
    measure and uses randomization to improve computational efficiency.

    Attributes:
        k (int): The user-expected number of final clusters.
        k_initial (int): The number of initial centroids to select (2*k).
        similarity_threshold (float): The dynamic threshold (lambda) used for merging clusters.
        centroids_ (List[Set[int]]): The documents representing each final cluster.
        clusters_ (List[Set[int]]): The final soft clusters, containing all member document indices.
        cluster_words_ (List[Set[str]]): The keywords representing each final cluster.
    """

    def __init__(self, k: int):
        """
        Initializes the SISC algorithm with the expected number of clusters.

        Args:
            k (int): The expected number of final clusters.
        """
        self.k = k
        self.k_initial = 2 * k  # Start with twice the expected clusters to be more robust
        self.similarity_threshold = 0.0
        self.centroids_: List[Set[int]] = []
        self.clusters_: List[Set[int]] = []
        self.cluster_words_: List[Set[str]] = []

    def _preprocess(self, docs: List[str]) -> List[Dict[str, int]]:
        """
        Transforms documents into a simple bag-of-words representation by
        removing common English stop words and converting to lowercase.
        
        Args:
            docs (List[str]): A list of text documents.
            
        Returns:
            List[Dict[str, int]]: A list of dictionaries, where each dictionary
                                  represents a document and maps words to their counts.
        """
        # A basic set of common English stop words.
        stop_words = set([
            "the", "a", "an", "and", "or", "but", "in", "on", "with", "for",
            "to", "of", "from", "at", "by", "is", "are", "be", "was", "were",
            "it", "its", "that", "this", "these", "those"
        ])

        processed_docs = []
        for doc in docs:
            words = doc.lower().split()
            word_counts = defaultdict(int)
            for word in words:
                if word not in stop_words:
                    word_counts[word] += 1
            processed_docs.append(word_counts)
        return processed_docs

    @staticmethod
    def _tanimoto_similarity(doc1_words: Dict[str, int], doc2_words: Dict[str, int]) -> float:
        """
        Calculates the Tanimoto similarity between two documents based on their
        bag-of-words representation. This serves as the core similarity measure.

        Formula: |A intersect B| / |A union B|

        Args:
            doc1_words (Dict[str, int]): Bag-of-words for the first document.
            doc2_words (Dict[str, int]): Bag-of-words for the second document.

        Returns:
            float: The Tanimoto similarity score (0.0 to 1.0).
        """
        set1 = set(doc1_words.keys())
        set2 = set(doc2_words.keys())
        n1 = len(set1)
        n2 = len(set2)
        m = len(set1.intersection(set2))
        
        denominator = n1 + n2 - m
        if denominator == 0:
            return 0.0
        return m / denominator

    def _initialize_centroids(self, processed_docs: List[Dict[str, int]]) -> List[Set[int]]:
        """
        Initializes cluster centroids by selecting documents that are far from
        each other. It also dynamically determines the similarity threshold (lambda)
        based on the dataset.

        Args:
            processed_docs (List[Dict[str, int]]): The pre-processed documents.

        Returns:
            List[Set[int]]: A list of initial cluster centroids (as sets of document indices).
        """
        num_docs = len(processed_docs)
        if num_docs == 0:
            return []

        doc_indices = list(range(num_docs))
        
        # Step 1: Pick k_initial centroids that are far from each other
        centroids_indices = set()
        first_centroid_idx = random.choice(doc_indices)
        centroids_indices.add(first_centroid_idx)

        remaining_docs = list(set(doc_indices) - centroids_indices)
        while len(centroids_indices) < self.k_initial and remaining_docs:
            max_min_similarity = -1
            best_doc_idx = -1
            
            for doc_idx in remaining_docs:
                min_similarity_to_centroids = float('inf')
                for centroid_idx in centroids_indices:
                    sim = self._tanimoto_similarity(processed_docs[doc_idx], processed_docs[centroid_idx])
                    min_similarity_to_centroids = min(min_similarity_to_centroids, sim)
                
                if min_similarity_to_centroids > max_min_similarity:
                    max_min_similarity = min_similarity_to_centroids
                    best_doc_idx = doc_idx
            
            if best_doc_idx != -1:
                centroids_indices.add(best_doc_idx)
                remaining_docs.remove(best_doc_idx)
            else:
                break
        
        # Step 2: Calculate all similarities to find the dynamic threshold (lambda)
        all_similarities = []
        for p_idx in centroids_indices:
            for q_idx in range(num_docs):
                all_similarities.append(self._tanimoto_similarity(processed_docs[p_idx], processed_docs[q_idx]))

        all_similarities.sort(reverse=True)
        
        # The threshold is set such that at least half of the documents are close
        # to at least one centroid.
        target_index = num_docs // 2
        if target_index < len(all_similarities):
            self.similarity_threshold = all_similarities[target_index]
        else:
            self.similarity_threshold = 0.0

        # Step 3: Filter out any outlier centroids
        initial_centroids = [{p} for p in centroids_indices]
        valid_centroids = []
        for centroid in initial_centroids:
            is_outlier = True
            centroid_doc_idx = list(centroid)[0]
            for doc_idx in range(num_docs):
                if self._tanimoto_similarity(processed_docs[centroid_doc_idx], processed_docs[doc_idx]) >= self.similarity_threshold:
                    is_outlier = False
                    break
            if not is_outlier:
                valid_centroids.append(centroid)

        return valid_centroids

    def _calculate_membership_measure(self, doc_idx: int, centroid: Set[int], processed_docs: List[Dict[str, int]]) -> float:
        """
        Calculates m(c, x), the average similarity of document x to a cluster centroid c.

        Args:
            doc_idx (int): The index of the document.
            centroid (Set[int]): The set of document indices in the cluster centroid.
            processed_docs (List[Dict[str, int]]): The pre-processed documents.
            
        Returns:
            float: The membership measure.
        """
        if not centroid:
            return 0.0
        
        total_similarity = sum(
            self._tanimoto_similarity(processed_docs[doc_idx], processed_docs[c_idx])
            for c_idx in centroid
        )
        return total_similarity / len(centroid)

    def _merge_clusters(self, centroids: List[Set[int]]) -> List[Set[int]]:
        """
        Merges similar clusters based on set relationships or document overlap.
        
        Args:
            centroids (List[Set[int]]): The list of current cluster centroids.
            
        Returns:
            List[Set[int]]: The merged list of centroids.
        """
        merged_centroids = []
        i = 0
        while i < len(centroids):
            j = i + 1
            merged_with_i = False
            while j < len(centroids):
                centroid_i = centroids[i]
                centroid_j = centroids[j]
                
                intersection_size = len(centroid_i.intersection(centroid_j))
                
                # Merge if one is a subset of the other or if there is significant overlap
                if (centroid_i.issubset(centroid_j) or 
                    centroid_j.issubset(centroid_i) or
                    intersection_size >= len(centroid_i) / 2 or
                    intersection_size >= len(centroid_j) / 2):
                    
                    centroids[i] = centroid_i.union(centroid_j)
                    del centroids[j]
                    merged_with_i = True
                else:
                    j += 1
            
            if not merged_with_i:
                merged_centroids.append(centroids[i])
                i += 1
            else:
                # If a merge happened, restart the inner loop for the new, larger cluster
                pass
                
        return merged_centroids

    def _extract_keywords(self, docs: List[str], final_clusters: List[Set[int]]) -> List[Set[str]]:
        """
        Extracts representative keywords for each cluster by collecting all
        words from the documents within each final cluster.
        
        Args:
            docs (List[str]): The original list of text documents.
            final_clusters (List[Set[int]]): The final clusters from the algorithm.
            
        Returns:
            List[Set[str]]: A list of sets, where each set contains the keywords
                            for a corresponding cluster.
        """
        cluster_keywords = []
        for cluster_docs in final_clusters:
            all_cluster_words = set()
            for doc_idx in cluster_docs:
                all_cluster_words.update(docs[doc_idx].lower().split())
            cluster_keywords.append(all_cluster_words)
        return cluster_keywords

    def fit_predict(self, docs: List[str]) -> csr_matrix:
        """
        Runs the SISC algorithm on a collection of documents.

        Args:
            docs (List[str]): A list of text documents.

        Returns:
            csr_matrix: A sparse matrix of shape (num_docs, num_clusters) indicating
                        document membership in the final clusters.
        """
        processed_docs = self._preprocess(docs)
        num_docs = len(docs)
        
        # Phase 1: Initialize centroids and similarity threshold
        centroids = self._initialize_centroids(processed_docs)

        # Phase 2: Iteratively refine and merge clusters
        while True:
            changes_made = False
            new_centroids = [set() for _ in range(len(centroids))]
            
            # Update each centroid's documents
            for c_idx, centroid in enumerate(centroids):
                for doc_idx in range(num_docs):
                    # Randomization step: recalculate with a probability
                    m_value = self._calculate_membership_measure(doc_idx, centroid, processed_docs)
                    if random.random() < m_value / (self.similarity_threshold + 1e-9):
                        if m_value >= self.similarity_threshold:
                            new_centroids[c_idx].add(doc_idx)
            
            # Check for significant changes in centroids to determine if the loop should terminate
            current_centroids_sets = [frozenset(c) for c in centroids]
            new_centroids_sets = [frozenset(c) for c in new_centroids]
            if len(current_centroids_sets) != len(new_centroids_sets) or set(current_centroids_sets) != set(new_centroids_sets):
                 changes_made = True
            
            if not changes_made:
                break
            
            centroids = [s for s in new_centroids if s]
            centroids = self._merge_clusters(centroids)
            
            if not centroids:
                break

        # Final clusters are the converged centroids
        self.clusters_ = centroids
        self.cluster_words_ = self._extract_keywords(docs, self.clusters_)
        
        # Phase 3: Prepare the output membership matrix
        num_final_clusters = len(self.clusters_)
        memberships = lil_matrix((num_docs, num_final_clusters), dtype=np.int8)
        
        for cluster_idx, doc_set in enumerate(self.clusters_):
            for doc_idx in doc_set:
                memberships[doc_idx, cluster_idx] = 1
                
        return memberships.tocsr()