# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""EVCLUS: evidential clustering of proximity data.

EVCLUS fits a mass function per object over the singletons plus the whole
frame, so an ambiguous object can put its mass on "I do not know" rather than
being forced to split it between clusters. This example fits three blobs plus
one deliberately ambiguous point and shows where the ignorance goes.

Run:
    python example/example_evclus.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from soft_clustering import EVCLUS  # noqa: E402


def blobs_with_an_ambiguous_point(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = np.vstack([rng.normal(c, 0.5, (n_per, 2)) for c in centres])
    y = np.repeat([0, 1, 2], n_per)
    # A point at the centroid of the three blobs: no cluster owns it.
    return np.vstack([X, centres.mean(axis=0)]), np.append(y, -1)


def main() -> None:
    X, y = blobs_with_an_ambiguous_point()

    model = EVCLUS(n_clusters=3, n_init=3, random_state=0).fit(X)

    print(f"credal partition masses_ {model.masses_.shape}")
    print("  columns 0..2 are the singleton masses, column 3 is m(Omega)")
    print(f"  rows sum to 1: {np.allclose(model.masses_.sum(axis=1), 1.0)}")
    print(f"pignistic memberships_ {model.memberships_.shape}")
    print(f"stress after fitting: {model.stress_:.5f}")

    ignorance = model.ignorance()
    print("\nMass on the whole frame (ignorance):")
    print(f"  median over the blob points : {np.median(ignorance[:-1]):.3f}")
    print(f"  the ambiguous point         : {ignorance[-1]:.3f}")
    print(
        "\nThe ambiguous point keeps more of its mass uncommitted, which is\n"
        "information a probability vector cannot express: a uniform (1/3, 1/3,\n"
        "1/3) row cannot be told apart from a point that genuinely sits\n"
        "between three clusters."
    )

    # EVCLUS is defined on proximities, so a dissimilarity matrix works too.
    squared = (X**2).sum(axis=1)
    distances = np.sqrt(
        np.maximum(squared[:, None] + squared[None, :] - 2 * X @ X.T, 0.0)
    )
    from_distances = EVCLUS(
        n_clusters=3, metric="precomputed", n_init=3, random_state=0
    ).fit(distances)
    print(
        "\nSame result from a precomputed dissimilarity matrix: "
        f"{np.allclose(model.memberships_, from_distances.memberships_, atol=1e-6)}"
    )

    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return
    mask = y >= 0
    print(
        f"\nARI against the generating partition: "
        f"{adjusted_rand_score(y[mask], model.labels_[mask]):.3f}"
    )


if __name__ == "__main__":
    main()
