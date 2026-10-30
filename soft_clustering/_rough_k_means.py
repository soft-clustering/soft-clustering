# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_array, check_generator, check_n_clusters, data_scale


class RoughKMeans(BaseSoftClusterer):
    """Rough k-means clustering (Lingras and West, 2004).

    Every cluster has a lower approximation (objects that certainly belong to
    it) and an upper approximation (objects that possibly do); the method,
    its parameters and its references are described under ``__init__``
    below. ``lower_approx_`` and ``upper_approx_`` hold the approximations.
    """

    # Under the Lingras-West assignment rule every object enters at least one
    # upper approximation, so the membership row read off the approximations
    # always normalises. This was False while the implementation could leave
    # an object in no approximation at all.
    _partition_constrained = True

    @typechecked
    def __init__(
        self,
        n_clusters: int = 2,
        weight_lower: float = 0.7,
        threshold: float = 1.2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | np.random.Generator | None = None,
    ):
        """
        Rough k-means clustering with interval-set (lower/upper) approximations.

        Each cluster has a lower approximation (objects that certainly belong
        to it) and an upper approximation (objects that possibly do). A
        centroid is ``weight_lower`` times the mean of its lower
        approximation plus ``1 - weight_lower`` times the mean of its
        boundary region (Lingras and West, 2004). An object is a boundary
        object when another centroid is within ``threshold`` times the
        distance to its nearest centroid; this relative threshold is the
        refinement of Peters (2006), which makes the rule scale-invariant.
        ``memberships_`` gives an object in a lower approximation membership
        one in that cluster and shares a boundary object equally among the
        clusters whose upper approximations contain it.

        References
        ----------
        P. Lingras and C. West. Interval set clustering of web users with
        rough k-means. *Journal of Intelligent Information Systems*,
        23(1):5--16, 2004. doi:10.1023/B:JIIS.0000029668.88665.1a

        G. Peters. Some refinements of rough k-means clustering. *Pattern
        Recognition*, 39(8):1481--1491, 2006.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters.
        weight_lower : float
            Mixing weight for lower vs. upper when updating centroids.
        threshold : float
            Relative boundary threshold ``>= 1``. An object goes in the
            boundary region of every cluster whose centroid is within
            ``threshold`` times the distance to its nearest centroid; with
            ``threshold = 1`` no object is ever in a boundary region and the
            method reduces to k-means.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance for centroid changes, relative to the
            data spread (root mean per-feature standard deviation).
        random_state : int, optional
            Random seed for reproducibility.
        """
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if not 0.0 <= weight_lower <= 1.0:
            raise ValueError(f"weight_lower must be in [0, 1], got {weight_lower}")
        if threshold < 1.0:
            raise ValueError(f"threshold must be >= 1, got {threshold}")

        self.n_clusters = n_clusters
        self.weight_lower = weight_lower
        self.threshold = threshold
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two points."""
        return np.linalg.norm(a - b)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Rough K-Means clustering on the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix where each row is a sample and each column is a feature.

        Returns
        -------
        memberships : ndarray of shape (n_samples, n_clusters)
            Lower-approximation objects belong fully to one cluster; boundary
            objects share membership equally across their upper approximations.
            The approximations, centers and iteration count remain available
            as ``lower_approx_``, ``upper_approx_``, ``centers_`` and ``n_iter_``.
        """
        name = type(self).__name__
        X = check_array(X, estimator=name)
        n_samples, _ = X.shape
        check_n_clusters(self.n_clusters, n_samples, estimator=name)

        # Initialize centroids using random samples
        rng = check_generator(self.random_state)
        initial_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[initial_idx].astype(float)

        # Initialize approximation sets
        L: list[set[int]] = [set() for _ in range(self.n_clusters)]  # Lower
        U: list[set[int]] = [set() for _ in range(self.n_clusters)]  # Upper

        iter_count = 0
        scale = data_scale(X)
        for iteration in range(self.max_iter):
            iter_count = iteration + 1
            old_centroids = centroids.copy()

            # Assignment rule of Lingras and West (2004, Sec. 3).
            #
            # For each object, let v be its nearest centroid and d_v that
            # distance. Every centroid within `threshold * d_v` forms the
            # object's boundary set T. If T holds more than just v the object
            # is a boundary object: it enters the upper approximation of every
            # member of T and the lower approximation of none. Otherwise it is
            # unambiguous and enters both approximations of v.
            #
            # The invariants this guarantees, and which
            # tests/test_external_agreement.py checks, are that L(j) is a
            # subset of U(j), that an object lies in at most one lower
            # approximation, and that every object lies in at least one upper
            # approximation. A previous implementation used absolute alpha and
            # beta radii derived from the inter-centroid distances, under
            # which an object far from every centroid entered no
            # approximation at all and so received an all-zero membership row.
            diff = X[:, None, :] - centroids[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)

            for j in range(self.n_clusters):
                L[j].clear()
                U[j].clear()

            nearest = np.argmin(dist_matrix, axis=1)
            nearest_dist = dist_matrix[np.arange(n_samples), nearest]
            # Guard the ratio for an object sitting exactly on a centroid.
            safe = np.maximum(nearest_dist, np.finfo(float).tiny)
            within = dist_matrix <= self.threshold * safe[:, None]

            for i in range(n_samples):
                companions = np.flatnonzero(within[i])
                if companions.size > 1:
                    for j in companions.tolist():
                        U[j].add(i)
                else:
                    j = int(nearest[i])
                    L[j].add(i)
                    U[j].add(i)

            # Update centroids using weighted average of approximations
            new_centroids = np.zeros_like(centroids)
            for j in range(self.n_clusters):
                lower_idxs = list(L[j])
                fringe_idxs = [i for i in U[j] if i not in L[j]]

                if lower_idxs and fringe_idxs:
                    mu_L = X[lower_idxs].mean(axis=0)
                    mu_F = X[fringe_idxs].mean(axis=0)
                    new_centroids[j] = (
                        self.weight_lower * mu_L + (1 - self.weight_lower) * mu_F
                    )
                elif lower_idxs:
                    new_centroids[j] = X[lower_idxs].mean(axis=0)
                elif fringe_idxs:
                    new_centroids[j] = X[fringe_idxs].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            centroids = new_centroids

            # Check convergence using centroid shifts
            shift = np.linalg.norm(centroids - old_centroids)
            if shift < self.tol * scale:
                break

        # Convert approximation sets to binary matrices
        lower_matrix = np.zeros((n_samples, self.n_clusters), dtype=int)
        upper_matrix = np.zeros((n_samples, self.n_clusters), dtype=int)
        for j in range(self.n_clusters):
            for i in L[j]:
                lower_matrix[i, j] = 1
            for i in U[j]:
                upper_matrix[i, j] = 1

        # Rough k-means yields interval sets rather than degrees. The estimator
        # protocol requires a membership matrix, so we read one off the
        # approximations: an object in a lower approximation belongs to that
        # cluster alone, while an object lying only in upper approximations is
        # shared uniformly among them (Lingras and West, 2004, Sec. 3).
        region = np.where(
            lower_matrix.sum(axis=1, keepdims=True) > 0, lower_matrix, upper_matrix
        ).astype(float)
        row_sums = region.sum(axis=1, keepdims=True)
        self.memberships_ = region / np.maximum(row_sums, 1.0)
        self.centers_ = centroids

        # Retain the interval sets alongside the dense memberships.
        self.lower_approx_ = lower_matrix
        self.upper_approx_ = upper_matrix
        self.n_iter_ = iter_count

        return self.memberships_
