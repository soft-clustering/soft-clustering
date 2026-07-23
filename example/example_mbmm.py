from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import MBMM

    np.random.seed(0)

    # Generate Beta data for 2 clusters
    X1 = np.random.beta(2, 5, size=(50, 2))
    X2 = np.random.beta(5, 2, size=(50, 2))
    X = np.vstack([X1, X2])

    model = MBMM(n_components=2, max_iter=100)
    model.fit(X)

    labels = model.predict()
    probs = model.predict_proba()

    print("Predicted Labels:")
    print(labels)

    print("\nMembership Probabilities:")
    print(probs)
