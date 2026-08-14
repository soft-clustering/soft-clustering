# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Gath--Geva clustering on anisotropic Gaussian clusters.

Gath--Geva replaces the Euclidean distance of fuzzy c-means with an
exponential distance built from each cluster's own fuzzy covariance and prior.
That is what lets it follow elongated clusters that fuzzy c-means, which
assumes spherical ones, cuts across. This example makes the difference
visible.

Run:
    python example/example_gathgeva.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from soft_clustering import FCM, GathGeva  # noqa: E402


def anisotropic_clusters(n_per=150, seed=0):
    """Two elongated clusters whose principal axes are orthogonal."""
    rng = np.random.default_rng(seed)
    first = rng.normal(0, 1, (n_per, 2)) @ np.array([[3.0, 0.0], [0.0, 0.25]])
    second = rng.normal(0, 1, (n_per, 2)) @ np.array([[0.25, 0.0], [0.0, 3.0]])
    second += np.array([4.0, 4.0])
    return np.vstack([first, second]), np.repeat([0, 1], n_per)


def main() -> None:
    X, y = anisotropic_clusters()

    model = GathGeva(n_clusters=2, m=2.0, random_state=0).fit(X)
    U = model.memberships_

    print(f"memberships_ {U.shape}, rows sum to 1: {np.allclose(U.sum(axis=1), 1.0)}")
    print(f"converged after {model.n_iter_} iterations")
    print(f"cluster priors: {np.round(model.priors_, 3)}")
    print("\nfuzzy covariance of cluster 0:")
    print(np.round(model.covariances_[0], 3))

    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return

    fcm = FCM(n_clusters=2, m=2.0, random_state=0).fit(X)
    print("\nAgreement with the generating partition (ARI):")
    print(f"  Gath--Geva      {adjusted_rand_score(y, model.labels_):.3f}")
    print(f"  fuzzy c-means   {adjusted_rand_score(y, fcm.labels_):.3f}")
    print(
        "\nGath--Geva models a full covariance per cluster, so it follows the\n"
        "elongated shapes; fuzzy c-means assumes spherical clusters and splits\n"
        "them across the wrong axis."
    )


if __name__ == "__main__":
    main()
