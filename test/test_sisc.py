from os import path
from scipy.sparse import csr_matrix
import sys
import os

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import SISC
    # A sample collection of documents to test the algorithm
    documents = [
        "the cat sat on the mat",
        "a dog ate my homework",
        "the cat and the dog are friends",
        "my homework is about machine learning",
        "machine learning models need data",
        "cat and dog are common pets"
    ]

    # Create an instance of the SISC class, expecting 2 final clusters.
    # The algorithm might produce more or fewer based on the data.
    sisc_model = SISC(k=2)

    # Run the clustering algorithm
    try:
        memberships_matrix = sisc_model.fit_predict(documents)
    except Exception as e:
        print(f"An error occurred during clustering: {e}")
        memberships_matrix = csr_matrix((len(documents), 0))
    
    print("--- SISC Clustering Results ---")
    print(f"Number of documents: {memberships_matrix.shape[0]}")
    print(f"Number of final clusters: {memberships_matrix.shape[1]}")
    
    print("\n--- Cluster Details ---")
    if memberships_matrix.shape[1] > 0:
        # Display the representative words and members of each cluster
        for i, words in enumerate(sisc_model.cluster_words_):
            cluster_docs_indices = sorted(list(sisc_model.clusters_[i]))
            print(f"Cluster {i+1}:")
            print(f"  Representative Words: {sorted(list(words))}")
            print(f"  Document Indices: {cluster_docs_indices}")
            print("-" * 20)
            
        print("\n--- Membership Matrix (Sparse) ---")
        print(memberships_matrix.toarray())
    else:
        print("No clusters were formed. This may be due to the nature of the input documents or the algorithm's parameters.")
        print("Final Centroids:", sisc_model.centroids_)
        print("Final Threshold:", sisc_model.similarity_threshold)
        
