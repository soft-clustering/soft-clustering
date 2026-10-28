# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Soft hybrid bipartite graph formulation (sHBGF) consensus."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._consensus import check_ensemble, one_hot
from ._validation import check_n_clusters, int_seed


@typechecked
class SHBGF(BaseSoftClusterer):
    r"""Soft HBGF consensus clustering of an ensemble of soft partitions.

    sHBGF (Punera and Ghosh, 2008) builds a bipartite graph between samples
    and the clusters of all input partitions, with edge weights given by the
    memberships, and partitions it. SCPP row-normalises the concatenated
    membership matrix (each sample's edge weights sum to one) and partitions
    the samples with k-means, a relaxation of the bipartite graph partitioning
    step (the reference uses METIS). The consensus is crisp, and
    ``memberships_`` is its one-hot encoding.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of consensus clusters.
    max_iter : int, default=300
        Maximum k-means iterations per run.
    n_init : int, default=10
        k-means restarts.
    random_state : int, numpy Generator or None, default=None
        Seed for k-means. Earlier releases fixed it at 42.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)

    References
    ----------
    K. Punera and J. Ghosh. Consensus-based ensembles of soft clusterings.
    *Applied Artificial Intelligence*, 22(7--8):780--810, 2008.
    doi:10.1080/08839510802170546
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, soft_memberships) -> np.ndarray:
        """Fit on a list of ``(n_samples, n_clusters_q)`` membership matrices."""
        name = type(self).__name__
        mats = check_ensemble(soft_memberships, name)
        K = check_n_clusters(self.n_clusters, mats[0].shape[0], estimator=name)
        Z = np.concatenate(mats, axis=1)
        Z /= Z.sum(axis=1, keepdims=True) + 1e-12
        labels = KMeans(
            n_clusters=K,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=int_seed(self.random_state),
        ).fit_predict(Z)
        self.consensus_labels_ = labels
        self.memberships_ = one_hot(labels, K)
        return self.memberships_
