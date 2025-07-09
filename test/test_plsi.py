from os import path
import sys
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import PLSI  
    
    # Example 1: Simple list of documents
    docs = [
        "apple banana apple",
        "banana fruit apple",
        "fruit banana banana"
    ]
    model1 = PLSI(n_topics=2, max_iter=30, tempered=True, random_state=42)
    model1.fit_predict(docs)
    print("Example 1: Perplexity:", model1.perplexity)
    print("P(w|d):\n", model1.get_P_w_given_d())

    # Example 2: Different number of topics, disable tempering
    model2 = PLSI(n_topics=3, max_iter=50, tempered=False, random_state=1)
    model2.fit_predict(docs)
    print("\nExample 2: Perplexity (EM only):", model2.perplexity)

    # Example 3: Sparse matrix input
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    model3 = PLSI(n_topics=2, max_iter=25, tempered=True, random_state=0)
    model3.fit_predict(X)
    print("\nExample 3: Vocabulary size:", len(vectorizer.get_feature_names_out()))
    print("Log Likelihoods:", model3.log_likelihoods)
