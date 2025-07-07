from os import path
import sys
import numpy as np
from scipy.sparse import csr_matrix

if __name__ == '__main__':
  base_dir = path.dirname(path.realpath(__file__))
  sys.path.append(base_dir[:-4])
  from soft_clustering import LDA

  # Example 1: Small raw-text corpus
  print("Example 1: Small Raw Text Corpus")
  docs = [
      "apple banana apple",
      "banana fruit apple",
      "fruit banana banana"
  ]
  model1 = LDA(n_topics=2, max_iter=20, var_max_iter=50, beta=0.1)
  model1.fit_predict(docs)
  print("Topic–word distributions:")
  model1.print_top_words(n_top_words=3)
  print("Document–topic mixtures (gamma):")
  print(model1.gamma)
  print("\n" + "-"*50 + "\n")
  
  # Example 2: Synthetic three-topic documents (Cats, Dogs, Birds)
  print("Example 2: Synthetic 'Cats vs Dogs vs Birds' Corpus")
  # Generate 5 docs about each animal
  cat_docs = ["cat feline kitty" * 5 for _ in range(5)]
  dog_docs = ["dog canine puppy" * 5 for _ in range(5)]
  bird_docs = ["bird avian feather" * 5 for _ in range(5)]
  mixed_docs = ["cat dog bird feline canine feather"]
  docs2 = cat_docs + dog_docs + bird_docs + mixed_docs
  model2 = LDA(n_topics=3, alpha=0.5, beta=0.1, max_iter=30, var_max_iter=100)
  model2.fit_predict(docs2)
  print("Topic–word distributions:")
  model2.print_top_words(n_top_words=3)
  print("Document–topic mixtures (gamma):")
  print(np.round(model2.gamma, 2))
  print("\n" + "-"*50 + "\n")
  
  # Example 3: Precomputed count matrix with known vocabulary
  print("Example 3: Precomputed Count Matrix")
  # Construct simple D=3, V=4 matrix
  counts = np.array([
      [4, 1, 0, 0],  # Document 0: mostly word0
      [0, 2, 3, 1],  # Document 1: mixed word1, word2, word3
      [1, 0, 4, 2]   # Document 2: mostly word2, some word3, word0
  ])
  vocab = ["apple", "banana", "cherry", "date"]
  X_csr = csr_matrix(counts)
  model3 = LDA(n_topics=2, alpha=1.0, beta=0.01, max_iter=25, var_max_iter=50)
  model3.fit_predict(X_csr, vocabulary=vocab)
  print("Topic–word distributions:")
  model3.print_top_words(n_top_words=4)
  print("Document–topic mixtures (gamma):")
  print(np.round(model3.gamma, 2))
