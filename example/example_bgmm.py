from os import path
import sys
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import BGMM

    np.random.seed(0)

    Xg = np.concatenate([
        np.random.normal(0, 1, 50),
        np.random.normal(5, 1, 50)
    ])

    Xb = np.concatenate([
        np.random.beta(2, 5, 50),
        np.random.beta(5, 2, 50)
    ])

    model = BGMM(n_components=2, max_iter=100)
    model.fit(Xg, Xb)

    probs = model.predict_proba()
    labels = model.predict()

    print("Predicted Labels:")
    print(labels)

    print("\nMembership Probabilities:")
    print(probs)
