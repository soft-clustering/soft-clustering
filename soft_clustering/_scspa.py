# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Soft cluster-based similarity partitioning (sCSPA) consensus."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._consensus import check_ensemble, one_hot
from ._validation import check_n_clusters, int_seed


@typechecked
class SCSPA(BaseSoftClusterer):
    r"""Soft CSPA consensus clustering of an ensemble of soft partitions.

    Each sample is represented by the concatenation of its membership
    vectors across the :math:`r` input partitions,
    :math:`\tilde u_i = [u_i^{(1)}, \dots, u_i^{(r)}]`. In sCSPA (Punera and
    Ghosh, 2008) the similarity of two samples is derived from these vectors
    and the resulting similarity graph is partitioned. SCPP L2-normalises the
    concatenated vectors --- so that their inner product is the cosine
    similarity the graph would use --- and partitions them with k-means, a
    standard relaxation of the graph partitioning step (the reference uses
    METIS). The consensus is crisp, and ``memberships_`` is its one-hot
    encoding.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of consensus clusters.
    n_init : int, default=10
        k-means restarts.
    random_state : int, numpy Generator or None, default=None
        Seed for k-means. Earlier releases fixed it at 42.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        One-hot consensus partition.

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
        n_init: int = 10,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, soft_memberships) -> np.ndarray:
        """Fit on a list of ``(n_samples, n_clusters_q)`` membership matrices.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
            One-hot consensus memberships.
        """
        name = type(self).__name__
        mats = check_ensemble(soft_memberships, name)
        K = check_n_clusters(self.n_clusters, mats[0].shape[0], estimator=name)
        Z = np.concatenate(mats, axis=1)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        labels = KMeans(
            n_clusters=K, n_init=self.n_init, random_state=int_seed(self.random_state)
        ).fit_predict(Z)
        self.consensus_labels_ = labels
        self.memberships_ = one_hot(labels, K)
        return self.memberships_
