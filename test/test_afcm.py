import numpy as np
from soft_clustering._afcm._afcm import AFCM


def test_afcm_graph_embedding():
    # Generate synthetic data with 2 clusters
    N = 100
    D = 2
    X = np.vstack([
        np.random.normal(loc=[0, 0], scale=0.5, size=(N // 2, D)),
        np.random.normal(loc=[4, 4], scale=0.5, size=(N // 2, D)),
    ])

    model = AFCM(c=2, lambda_=1.0)
    labels, U = model.fit_predict(X)

    print("Consensus Cluster Labels:", labels)
    print("Membership Matrix U:", U)


if __name__ == "__main__":
    test_afcm_graph_embedding()
