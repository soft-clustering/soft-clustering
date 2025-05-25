from os import path
import sys

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import WBSC

    # A sample collection of documents
    documents = [
        "the cat sat on the mat",
        "a dog ate my homework",
        "the cat and the dog are friends",
        "my homework is about machine learning",
        "machine learning models need data",
        "cat and dog are common pets"
    ]

    # Create an instance of the WBSC class
    # Using a lower threshold to encourage more merges in this small example
    wbsc_model = WBSC(similarity_threshold=0.2, min_doc_freq=2)

    # Run the algorithm
    memberships_matrix = wbsc_model.fit_predict(documents)

    print("--- WBSC Clustering Results ---")
    print(f"Number of documents: {memberships_matrix.shape[0]}")
    print(f"Number of final clusters: {memberships_matrix.shape[1]}")
    print("\n--- Cluster Details ---")

    # Display the representative words and members of each cluster
    for i, words in enumerate(wbsc_model.cluster_words_):
        cluster_docs_indices = wbsc_model.clusters_[i]
        print(f"Cluster {i+1}:")
        print(f"  Representative Words: {sorted(list(words))}")
        print(f"  Document Indices: {sorted(list(cluster_docs_indices))}")
        print("-" * 20)

    # Display the membership matrix
    print("\n--- Membership Matrix (Sparse) ---")
    print(memberships_matrix)
