import numpy as np
from typeguard import typechecked
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix
from typing import List, Dict, Set, Union

@typechecked
class WBSC:
    """
    Implements the Word-Based Soft Clustering (WBSC) algorithm.

    This algorithm performs soft clustering on documents based on the words they share,
    meaning a single document can belong to multiple clusters. This is useful for
    capturing multiple themes within a document.

    Attributes:
        similarity_threshold (float): The similarity score required to merge two clusters.
        min_doc_freq (int): The minimum number of documents a word must appear in to be considered.
        max_doc_freq_ratio (float): The maximum ratio of documents a word can appear in to be considered.
        clusters_ (List[Set[int]]): A list of the final clusters. Each cluster is a set of document indices.
        cluster_words_ (List[Set[str]]): The representative words for each final cluster.
    """

    def __init__(self,
                 similarity_threshold: float = 0.33,
                 min_doc_freq: int = 2,
                 max_doc_freq_ratio: float = 0.5):
        """
        Initializes the algorithm's parameters.

        Args:
            similarity_threshold (float): The Tanimoto similarity threshold for merging clusters.
                                          The paper uses 0.33 as a working value.
            min_doc_freq (int): Words appearing in fewer than this number of documents will be ignored.
            max_doc_freq_ratio (float): Words appearing in more than this ratio of documents will be ignored.
        """
        self.similarity_threshold = similarity_threshold
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq_ratio = max_doc_freq_ratio

        self.clusters_: List[Set[int]] = []
        self.cluster_words_: List[Set[str]] = []

    def _initialize_clusters(self, docs: List[str]) -> List[Dict[str, Union[Set[int], Set[str]]]]:
        """
        Performs initial processing of documents, filters the vocabulary, and creates initial word-based clusters.
        """
        num_docs = len(docs)
        word_doc_counts = defaultdict(int)
        word_to_docs_map = defaultdict(set)

        # First pass: Count document frequencies and build word-to-document mapping
        for i, doc in enumerate(docs):
            # Simple tokenization
            words = set(doc.lower().split())
            for word in words:
                word_doc_counts[word] += 1
                word_to_docs_map[word].add(i)

        # Filter vocabulary based on document frequency
        max_doc_freq = int(num_docs * self.max_doc_freq_ratio)
        filtered_vocabulary = {
            word for word, count in word_doc_counts.items()
            if count >= self.min_doc_freq and count <= max_doc_freq
        }

        # Create an initial cluster for each word in the filtered vocabulary
        initial_clusters = [
            {'docs': word_to_docs_map[word], 'words': {word}}
            for word in filtered_vocabulary
        ]
        return initial_clusters

    @staticmethod
    def _calculate_tanimoto(set1: Set, set2: Set) -> float:
        """
        Calculates the Tanimoto similarity between two sets.
        This measure was settled on after trying different measures.
        Formula: |A intersect B| / |A union B|
        """
        intersection_size = len(set1.intersection(set2))
        if intersection_size == 0:
            return 0.0
        union_size = len(set1.union(set2))
        return intersection_size / union_size

    def fit_predict(self, docs: List[str]) -> csr_matrix:
        """
        Runs the clustering algorithm on a collection of documents.

        Args:
            docs (List[str]): A list of text documents.

        Returns:
            csr_matrix: A sparse matrix of shape (num_docs, num_clusters) indicating membership.
        """
        # Phase 1: Create initial clusters
        clusters = self._initialize_clusters(docs)

        # Phase 2: Hierarchically merge clusters
        while True:
            merged_in_pass = False
            i = 0
            while i < len(clusters):
                j = i + 1
                merged_indices = []
                
                # The base cluster for potential merges in this iteration
                base_cluster_docs = clusters[i]['docs']
                base_cluster_words = clusters[i]['words']

                while j < len(clusters):
                    other_cluster_docs = clusters[j]['docs']
                    
                    # Merge if one cluster is a subset of the other
                    is_subset = base_cluster_docs.issubset(other_cluster_docs) or other_cluster_docs.issubset(base_cluster_docs)
                    
                    # Or if Tanimoto similarity is above the threshold 
                    similarity = self._calculate_tanimoto(base_cluster_docs, other_cluster_docs)

                    if is_subset or similarity > self.similarity_threshold:
                        # Perform the merge
                        base_cluster_docs = base_cluster_docs.union(other_cluster_docs)
                        base_cluster_words = base_cluster_words.union(clusters[j]['words']) # New cluster acquires words from both
                        merged_indices.append(j)
                        merged_in_pass = True
                    
                    j += 1
                
                # If merges occurred, update the cluster list
                if merged_indices:
                    clusters[i] = {'docs': base_cluster_docs, 'words': base_cluster_words}
                    # Delete merged clusters from the end to avoid index errors
                    for index in sorted(merged_indices, reverse=True):
                        del clusters[index]
                i += 1
            
            # If a full pass results in no merges, the process is complete
            if not merged_in_pass:
                break
        
        # Phase 3: Prepare the output
        self.clusters_ = [c['docs'] for c in clusters]
        self.cluster_words_ = [c['words'] for c in clusters]
        
        num_docs = len(docs)
        num_final_clusters = len(self.clusters_)
        
        # Create a sparse membership matrix
        memberships = lil_matrix((num_docs, num_final_clusters), dtype=np.int8)
        for cluster_idx, doc_set in enumerate(self.clusters_):
            for doc_idx in doc_set:
                memberships[doc_idx, cluster_idx] = 1
                
        return memberships.tocsr()