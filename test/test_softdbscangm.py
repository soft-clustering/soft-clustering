import numpy as np
from soft_clustering._soft_dbscan_gm._soft_dbscan_gm import SoftDBSCANGM


def test_soft_dbscan_gm():
    from sklearn.datasets import make_moons

    # ساخت دیتا با شکل پیچیده (مثل ماه‌ها)
    X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)

    model = SoftDBSCANGM(eps=0.3, min_samples=5, m=2.0, max_iter=100)
    model.fit(X)

    labels = model.predict()
    U = model.get_membership()

    print("Predicted Labels:")
    print(labels)

    print("\nFuzzy Membership Matrix:")
    print(U)


if __name__ == "__main__":
    test_soft_dbscan_gm()
