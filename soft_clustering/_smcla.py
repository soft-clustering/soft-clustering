# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Soft meta-clustering algorithm (sMCLA) consensus."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._consensus import check_ensemble
from ._validation import check_n_clusters, int_seed


@typechecked
class SMCLA(BaseSoftClusterer):
    r"""Soft MCLA consensus clustering of an ensemble of soft partitions.

    sMCLA (Punera and Ghosh, 2008) clusters the *clusters*: every column of
    every input membership matrix is a vector over the samples, these
    vectors are grouped into ``n_clusters`` meta-clusters, and a sample's
    consensus membership in a meta-cluster is the (normalised) sum of its
    memberships in the clusters assigned to it. Unlike sCSPA and sHBGF, the
    consensus is therefore *soft*. SCPP groups the cluster vectors with
    k-means (the reference partitions a similarity graph of them with METIS).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of meta-clusters.
    n_init : int, default=10
    random_state : int, numpy Generator or None, default=None
        Seed for k-means. Earlier releases fixed it at 42.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Soft consensus memberships; rows sum to one (a sample with zero
        membership everywhere receives the uniform row).
    meta_labels_ : ndarray of shape (total_input_clusters,)
        Meta-cluster of each input cluster.

    References
    ----------
    K. Punera and J. Ghosh. Consensus-based ensembles of soft clusterings.
    *Applied Artificial Intelligence*, 22(7--8):780--810, 2008.
    doi:10.1080/08839510802170546

    Notes
    -----
    Earlier releases computed this soft consensus and then discarded it,
    returning only its arg-max as a one-hot matrix.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_clusters: int = 3,
        n_init: int = 10,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, soft_memberships) -> np.ndarray:
        """Fit on a list of ``(n_samples, n_clusters_q)`` membership matrices."""
        name = type(self).__name__
        mats = check_ensemble(soft_memberships, name)
        cluster_vectors = np.concatenate([M.T for M in mats], axis=0)  # (sum K_q, n)
        K = check_n_clusters(self.n_clusters, cluster_vectors.shape[0], estimator=name)
        meta = KMeans(
            n_clusters=K, n_init=self.n_init, random_state=int_seed(self.random_state)
        ).fit_predict(cluster_vectors)

        n = mats[0].shape[0]
        consensus = np.zeros((n, K))
        for k in range(K):
            consensus[:, k] = cluster_vectors[meta == k].sum(axis=0)
        row = consensus.sum(axis=1, keepdims=True)
        self.meta_labels_ = meta
        self.memberships_ = np.where(
            row > 0, consensus / np.where(row > 0, row, 1.0), 1.0 / K
        )
        return self.memberships_
