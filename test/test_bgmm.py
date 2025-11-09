import numpy as np
from soft_clustering._bgmm._bgmm import BGMM


def test_bgmm():
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


if __name__ == "__main__":
    test_bgmm()
