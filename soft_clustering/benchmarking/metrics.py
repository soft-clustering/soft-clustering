"""Cluster-validity metrics for hard and soft partitions.

Seven measures operate on the membership matrix :math:`U` and five on
defuzzified labels. Two properties of this module are worth stating up front,
because getting either wrong silently produces numbers that look plausible.

**Fuzziness indices are not quality scores.** The partition coefficient,
modified partition coefficient and partition entropy measure *how crisp* a
partition is, not how good it is. A fuzzy method that has degenerated to a
hard partition scores :math:`\\mathrm{PC} = 1` and :math:`\\mathrm{PE} = 0`,
the best attainable values, while telling you nothing about cluster
structure. Compare them within one algorithm across settings, not across
algorithms.

**They also require the partition constraint.** :math:`\\mathrm{PC} \\le 1`
holds only when the rows of :math:`U` sum to one. Possibilistic and
affiliation-based methods (PCM, BigCLAM, NOCD, ...) emit unnormalised
typicalities for which these three indices are not defined; the functions
below return ``nan`` for them rather than a number on a different scale.
:func:`soft_clustering_metrics` detects this from ``U`` itself, and
:attr:`~soft_clustering.BaseSoftClusterer._partition_constrained` records the
same fact declaratively on every estimator.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - scikit-learn is a hard dependency
    _HAVE_SKLEARN = False

#: Tolerance used to decide whether the rows of ``U`` sum to one.
PARTITION_TOL = 1e-6

_HARD_METRICS = ("silhouette", "calinski_harabasz", "davies_bouldin", "ari", "nmi")


def is_partition_constrained(U: np.ndarray, tol: float = PARTITION_TOL) -> bool:
    """``True`` when every row of ``U`` sums to one within ``tol``."""
    if U.size == 0:
        return False
    return bool(np.all(np.abs(U.sum(axis=1) - 1.0) <= tol))


# ============================================================
# Hard Clustering Metrics
# ============================================================


def clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Standard clustering metrics on defuzzified labels.

    Every key is always present. A metric that is undefined for the given
    input --- an internal index on a single-cluster partition, or an external
    index without ``y_true`` --- is reported as ``nan`` rather than omitted,
    so that a benchmark table has no ragged rows.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    y_true : ndarray of shape (n_samples,), optional

    Returns
    -------
    dict
        ``silhouette``, ``calinski_harabasz``, ``davies_bouldin``, ``ari``,
        ``nmi``.
    """
    if not _HAVE_SKLEARN:  # pragma: no cover - scikit-learn is a hard dependency
        raise ImportError(
            "scikit-learn is required for clustering_metrics; "
            "install it with: pip install scikit-learn"
        )

    results: dict[str, float] = dict.fromkeys(_HARD_METRICS, float("nan"))

    labels = np.asarray(labels)
    n_labels = len(np.unique(labels))
    n_samples = len(labels)
    # Internal indices are defined for 2 <= n_labels <= n_samples - 1. Both
    # ends occur in practice: a collapsed partition has one cluster, and a
    # density method that turns every noise point into its own component --
    # SoftDBSCANGM does -- can reach n_labels == n_samples.
    if 1 < n_labels < n_samples:
        results["silhouette"] = float(silhouette_score(X, labels))
        results["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        results["davies_bouldin"] = float(davies_bouldin_score(X, labels))

    if y_true is not None:
        results["ari"] = float(adjusted_rand_score(y_true, labels))
        results["nmi"] = float(normalized_mutual_info_score(y_true, labels))

    return results


# ============================================================
# Fuzziness indices (require the partition constraint)
# ============================================================


def partition_coefficient(U: np.ndarray) -> float:
    """Partition coefficient, :math:`\\mathrm{PC} = \\frac1n \\sum_{ik} u_{ik}^2`.

    Ranges over :math:`[1/K, 1]`, where :math:`1` is a crisp partition. This
    is a measure of crispness, not of cluster quality. Returns ``nan`` when
    the rows of ``U`` do not sum to one, for which the upper bound does not
    hold.
    """
    if not is_partition_constrained(U):
        return float("nan")
    return float(np.sum(U**2) / U.shape[0])


def partition_entropy(U: np.ndarray, eps: float = 1e-12) -> float:
    """Partition entropy, :math:`\\mathrm{PE} = -\\frac1n \\sum_{ik} u_{ik}\\log u_{ik}`.

    Natural logarithm. Ranges over :math:`[0, \\log K]`, where :math:`0` is a
    crisp partition. Like the partition coefficient this measures crispness;
    returns ``nan`` for unnormalised memberships.
    """
    if not is_partition_constrained(U):
        return float("nan")
    return float(-np.sum(U * np.log(U + eps)) / U.shape[0])


def modified_partition_coefficient(U: np.ndarray) -> float:
    """Cluster-count-corrected partition coefficient,
    :math:`(\\mathrm{PC} - 1/K)/(1 - 1/K)`.

    Ranges over :math:`[0, 1]`. Returns ``nan`` for unnormalised memberships,
    and for :math:`K = 1`, where the correction is undefined.
    """
    n_clusters = U.shape[1]
    if n_clusters < 2:
        return float("nan")
    pc = partition_coefficient(U)
    if not np.isfinite(pc):
        return float("nan")
    return float((pc - 1.0 / n_clusters) / (1.0 - 1.0 / n_clusters))


# ============================================================
# Prototype-based validity indices
# ============================================================


def fuzzy_hypervolume(
    X: np.ndarray,
    U: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
) -> float:
    """Fuzzy hypervolume, :math:`\\mathrm{FHV} = \\sum_i \\sqrt{\\det F_i}`.

    :math:`F_i` is the membership-weighted fuzzy covariance of cluster
    :math:`i`,

    .. math::

        F_i = \\frac{\\sum_j u_{ij}^m (x_j - v_i)(x_j - v_i)^\\top}
                    {\\sum_j u_{ij}^m}.

    The index is the total volume the clusters occupy in feature space, so it
    is measured in :math:`(\\text{units of } X)^{d}` and only comparable
    between partitions of the *same* data with the same :math:`d`. Lower is
    more compact.

    Reference: I. Gath and A. B. Geva, *Fuzzy clustering for the estimation of
    the parameters of the components of mixtures of normal distributions*,
    Pattern Recognition Letters, 9(2):77--86, 1989.
    """
    X = np.asarray(X, dtype=np.float64)
    U_m = np.asarray(U, dtype=np.float64) ** m
    weights = U_m.sum(axis=0)

    diff = X[:, None, :] - np.asarray(centers, dtype=np.float64)[None, :, :]
    cov = np.einsum("nk,nkd,nke->kde", U_m, diff, diff, optimize=True)
    cov /= np.maximum(weights, np.finfo(float).tiny)[:, None, None]

    # A cluster confined to a subspace has determinant zero, which is a
    # legitimate value here (zero volume), so no ridge is added.
    determinants = np.maximum(np.linalg.det(cov), 0.0)
    return float(np.sum(np.sqrt(determinants)))


def xie_beni_index(
    X: np.ndarray,
    U: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
) -> float:
    """Xie--Beni index: compactness over minimum prototype separation.

    :math:`\\mathrm{XB} = \\sum_{ij} u_{ij}^m \\lVert x_j - v_i \\rVert^2 /
    (n \\min_{i \\neq l} \\lVert v_i - v_l \\rVert^2)`. Lower is better.
    """
    n_samples = X.shape[0]
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    numerator = np.sum((U**m) * (distances**2))

    center_distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    center_distances[center_distances == 0] = np.inf
    min_center_distance = np.min(center_distances)
    if not np.isfinite(min_center_distance):
        # Every prototype coincides; separation is zero and XB diverges.
        return float("inf")

    return float(numerator / (n_samples * min_center_distance**2))


def fuzzy_separation_index(centers: np.ndarray) -> float:
    """Average pairwise prototype separation. Higher is better."""
    if len(centers) < 2:
        return float("nan")
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    mask = ~np.eye(len(centers), dtype=bool)
    return float(np.mean(distances[mask]))


def fuzzy_compactness(
    X: np.ndarray,
    U: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
) -> float:
    """Membership-weighted within-cluster dispersion,
    :math:`\\sum_{ij} u_{ij}^m \\lVert x_j - v_i \\rVert^2`. Lower is better.

    This is the fuzzy c-means objective, so it decreases with :math:`K` by
    construction and should not be used to select the number of clusters.
    """
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return float(np.sum((U**m) * (distances**2)))


# ============================================================
# Unified Soft Evaluation
# ============================================================


def soft_clustering_metrics(
    X: np.ndarray,
    U: np.ndarray,
    centers: np.ndarray | None = None,
    m: float = 2.0,
    partition_constrained: bool | None = None,
) -> dict[str, float]:
    """Compute every applicable soft-clustering metric.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    U : ndarray of shape (n_samples, n_clusters)
        Membership matrix.
    centers : ndarray of shape (n_clusters, n_features), optional
        Required by the three prototype-based indices; they are reported as
        ``nan`` when it is absent.
    m : float, default=2.0
        Fuzzifier used to weight the memberships.
    partition_constrained : bool, optional
        Whether the rows of ``U`` are required to sum to one. Detected from
        ``U`` when omitted. Pass an estimator's
        ``_partition_constrained`` to state it explicitly. When ``False``, the
        three fuzziness indices are reported as ``nan``, because they are not
        defined on unnormalised memberships.

    Returns
    -------
    dict
        Seven keys, always present, ``nan`` where the metric does not apply.
    """
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    if partition_constrained is None:
        partition_constrained = is_partition_constrained(U)

    nan = float("nan")
    if partition_constrained:
        results = {
            "partition_coefficient": partition_coefficient(U),
            "modified_partition_coefficient": modified_partition_coefficient(U),
            "partition_entropy": partition_entropy(U),
        }
    else:
        results = {
            "partition_coefficient": nan,
            "modified_partition_coefficient": nan,
            "partition_entropy": nan,
        }

    results["fuzzy_hypervolume"] = nan
    results["xie_beni"] = nan
    results["fuzzy_compactness"] = nan
    results["fuzzy_separation"] = nan

    if centers is not None:
        centers = np.asarray(centers, dtype=np.float64)
        if centers.shape[0] == U.shape[1] and centers.shape[1] == X.shape[1]:
            results["fuzzy_hypervolume"] = fuzzy_hypervolume(X, U, centers, m)
            results["xie_beni"] = xie_beni_index(X, U, centers, m)
            results["fuzzy_compactness"] = fuzzy_compactness(X, U, centers, m)
            results["fuzzy_separation"] = fuzzy_separation_index(centers)

    return results
