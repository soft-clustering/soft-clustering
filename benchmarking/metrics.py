from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# ============================================================
# Hard Clustering Metrics
# ============================================================

def clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Standard clustering metrics.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    labels : ndarray of shape (n_samples,)

    y_true : ndarray, optional

    Returns
    -------
    dict
    """

    results = {}

    if len(np.unique(labels)) > 1:

        results["silhouette"] = silhouette_score(
            X,
            labels,
        )

        results["calinski_harabasz"] = (
            calinski_harabasz_score(
                X,
                labels,
            )
        )

        results["davies_bouldin"] = (
            davies_bouldin_score(
                X,
                labels,
            )
        )

    if y_true is not None:

        results["ari"] = adjusted_rand_score(
            y_true,
            labels,
        )

        results["nmi"] = normalized_mutual_info_score(
            y_true,
            labels,
        )

    return results


# ============================================================
# Soft Clustering Metrics
# ============================================================

def partition_coefficient(
    U: np.ndarray,
) -> float:
    """
    Partition Coefficient (PC).

    Higher is better.
    Maximum = 1.
    """

    n_samples = U.shape[0]

    return np.sum(U**2) / n_samples


def partition_entropy(
    U: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Partition Entropy (PE).

    Lower is better.
    """

    n_samples = U.shape[0]

    return -np.sum(
        U * np.log(U + eps)
    ) / n_samples


def modified_partition_coefficient(
    U: np.ndarray,
) -> float:
    """
    Modified Partition Coefficient (MPC).

    Corrects bias of PC with respect
    to number of clusters.

    Higher is better.
    """

    n_samples, n_clusters = U.shape

    pc = partition_coefficient(U)

    return (
        pc - (1.0 / n_clusters)
    ) / (
        1.0 - (1.0 / n_clusters)
    )


def fuzzy_hypervolume(
    U: np.ndarray,
) -> float:
    """
    Fuzzy Hypervolume.

    Lower values indicate
    more compact clusters.
    """

    return np.mean(
        np.prod(U, axis=1)
    )


# ============================================================
# Prototype-Based Metrics
# ============================================================

def xie_beni_index(
    X: np.ndarray,
    U: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
) -> float:
    """
    Xie-Beni Index.

    Lower is better.
    """

    n_samples = X.shape[0]

    distances = np.linalg.norm(
        X[:, None, :] - centers[None, :, :],
        axis=2,
    )

    numerator = np.sum(
        (U**m) * (distances**2)
    )

    center_distances = np.linalg.norm(
        centers[:, None, :]
        - centers[None, :, :],
        axis=2,
    )

    center_distances[
        center_distances == 0
    ] = np.inf

    min_center_distance = np.min(
        center_distances
    )

    denominator = (
        n_samples
        * min_center_distance**2
    )

    return numerator / denominator


def fuzzy_separation_index(
    centers: np.ndarray,
) -> float:
    """
    Average pairwise centroid separation.

    Higher is better.
    """

    distances = np.linalg.norm(
        centers[:, None, :]
        - centers[None, :, :],
        axis=2,
    )

    mask = ~np.eye(
        len(centers),
        dtype=bool,
    )

    return np.mean(
        distances[mask]
    )


def fuzzy_compactness(
    X: np.ndarray,
    U: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
) -> float:
    """
    Cluster compactness.

    Lower is better.
    """

    distances = np.linalg.norm(
        X[:, None, :]
        - centers[None, :, :],
        axis=2,
    )

    return np.sum(
        (U**m)
        * (distances**2)
    )


# ============================================================
# Unified Soft Evaluation
# ============================================================

def soft_clustering_metrics(
    X: np.ndarray,
    U: np.ndarray,
    centers: Optional[np.ndarray] = None,
    m: float = 2.0,
) -> Dict[str, float]:
    """
    Compute all soft clustering metrics.

    Parameters
    ----------
    X : ndarray

    U : ndarray
        Membership matrix.

    centers : ndarray, optional

    m : float, default=2.0

    Returns
    -------
    dict
    """

    results = {
        "partition_coefficient":
            partition_coefficient(U),

        "modified_partition_coefficient":
            modified_partition_coefficient(U),

        "partition_entropy":
            partition_entropy(U),

        "fuzzy_hypervolume":
            fuzzy_hypervolume(U),
    }

    if centers is not None:

        results["xie_beni"] = (
            xie_beni_index(
                X,
                U,
                centers,
                m,
            )
        )

        results["fuzzy_compactness"] = (
            fuzzy_compactness(
                X,
                U,
                centers,
                m,
            )
        )

        results["fuzzy_separation"] = (
            fuzzy_separation_index(
                centers
            )
        )

    return results