# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Soft kernel spectral clustering (Langone, Mall and Suykens, 2013)."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.linalg import eig
from sklearn.metrics.pairwise import rbf_kernel
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_array, check_n_clusters, check_n_features, check_scalar


def _soft_memberships(E: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Eq. (1.8): ``sm_iq = prod_{j != q} d_ij / sum_p prod_{j != p} d_ij``."""
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-300)
    Sn = prototypes / np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-300
    )
    D = np.clip(1.0 - En @ Sn.T, 0.0, 2.0)  # cosine distances, (n, k)
    k = D.shape[1]
    # prod_{j != q} d_ij, computed without dividing by a possibly-zero d_iq.
    prods = np.empty_like(D)
    for q in range(k):
        prods[:, q] = np.prod(np.delete(D, q, axis=1), axis=1)
    total = prods.sum(axis=1, keepdims=True)
    return np.where(total > 0, prods / np.where(total > 0, total, 1.0), 1.0 / k)


@typechecked
class SoftKSC(BaseSoftClusterer):
    r"""Soft kernel spectral clustering (SKSC).

    Kernel spectral clustering (KSC; Alzate and Suykens, 2010) is a weighted
    kernel PCA formulation of spectral clustering with an explicit,
    out-of-sample model. With the RBF kernel matrix :math:`\Omega`, its
    degree matrix :math:`D = \mathrm{diag}(\Omega \mathbf 1)` and the
    centring matrix :math:`M_D = I - \mathbf 1 \mathbf 1^\top D^{-1} /
    (\mathbf 1^\top D^{-1} \mathbf 1)`, the dual problem is

    .. math::

        D^{-1} M_D\, \Omega\, \alpha^{(l)} = \lambda_l \alpha^{(l)},
        \qquad l = 1, \dots, k - 1,

    keeping the :math:`k - 1` leading eigenvectors. Every point --- seen or
    unseen --- is projected as :math:`e^{(l)}(x) = \sum_j \alpha_j^{(l)}
    K(x_j, x) + b_l`, with :math:`b_l = -\mathbf 1^\top D^{-1} \Omega
    \alpha^{(l)} / \mathbf 1^\top D^{-1}\mathbf 1`. KSC assigns training
    points through an error-correcting output code: the :math:`k` most
    frequent sign patterns of the :math:`\alpha` rows form a codebook, and
    each point takes the code-word nearest in Hamming distance.

    SKSC (Langone, Mall and Suykens, 2013) turns that partition into soft
    memberships. The prototype of cluster :math:`p` is the mean projection
    :math:`s_p` of its points, the cosine distance of a point to it is
    :math:`d_{ip} = 1 - e_i^\top s_p / (\lVert e_i\rVert \lVert s_p\rVert)`,
    and the membership is

    .. math::

        \mathrm{sm}_i^{(q)} = \frac{\prod_{j \ne q} d_{ij}}
                                  {\sum_{p} \prod_{j \ne p} d_{ij}} ,

    which sums to one. Because the projection is an explicit function of
    :math:`x`, ``predict`` and ``predict_proba`` accept unseen data.

    **Cost.** Training solves a dense, non-symmetric ``n x n`` eigenproblem,
    :math:`O(n^3)` time and :math:`O(n^2)` memory. The reference trains on a
    subsample and extends to the rest out of sample; do the same for large
    data.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters :math:`k \ge 2`.
    gamma : float, default=1.0
        RBF kernel parameter, :math:`K(x, y) = \exp(-\gamma\lVert x -
        y\rVert^2)`, i.e. :math:`\gamma = 1/\sigma^2` in the reference's
        notation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    alphas_ : ndarray of shape (n_samples, n_clusters - 1)
        Dual eigenvectors :math:`\alpha^{(l)}`.
    bias_ : ndarray of shape (n_clusters - 1,)
    projections_ : ndarray of shape (n_samples, n_clusters - 1)
        Training projections :math:`e_i`.
    prototypes_ : ndarray of shape (n_clusters, n_clusters - 1)
        Cluster prototypes :math:`s_p` in the projection space.
    codebook_ : ndarray of shape (n_clusters, n_clusters - 1)

    References
    ----------
    R. Langone, R. Mall and J. A. K. Suykens. Soft kernel spectral
    clustering. In *Proc. International Joint Conference on Neural Networks
    (IJCNN)*, pp. 1--8, 2013. doi:10.1109/IJCNN.2013.6706850

    C. Alzate and J. A. K. Suykens. Multiway spectral clustering with
    out-of-sample extensions through weighted kernel PCA. *IEEE Transactions
    on Pattern Analysis and Machine Intelligence*, 32(2):335--347, 2010.

    Notes
    -----
    Earlier releases exposed under this name a two-class, semi-supervised
    kernel ridge classifier, ``fit(X_labeled, y_labeled, X_unlabeled)``,
    which is not the cited method. SKSC is unsupervised: ``fit(X)``.
    """

    _supports_out_of_sample = True
    _partition_constrained = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(self, n_clusters: int = 2, gamma: float = 1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma

    def fit(self, X, y=None, *legacy) -> SoftKSC:
        """Fit on a feature matrix of shape ``(n_samples, n_features)``.

        ``y`` is ignored, as for every unsupervised scikit-learn estimator.
        """
        name = type(self).__name__
        if legacy:
            raise TypeError(
                f"{name}.fit takes a single feature matrix. The semi-supervised "
                "signature fit(X_labeled, y_labeled, X_unlabeled) of earlier "
                "releases did not implement soft kernel spectral clustering and "
                "has been removed."
            )
        X = check_array(X, estimator=name, min_samples=2)
        n = X.shape[0]
        k = check_n_clusters(self.n_clusters, n, estimator=name)
        if k < 2:
            raise ValueError(f"{name}: n_clusters must be >= 2; got {k}.")
        gamma = check_scalar(
            self.gamma, "gamma", estimator=name, lower=0.0, lower_inclusive=False
        )

        Omega = rbf_kernel(X, gamma=gamma)
        d_inv = 1.0 / Omega.sum(axis=1)
        ones = np.ones(n)
        norm = float(ones @ d_inv)
        MD = np.eye(n) - np.outer(ones, d_inv) / norm
        A = d_inv[:, None] * (MD @ Omega)
        vals, vecs = eig(A)
        order = np.argsort(-vals.real)[: k - 1]
        alphas = np.real(vecs[:, order])
        bias = -(d_inv @ Omega @ alphas) / norm
        E = Omega @ alphas + bias[None, :]

        # ECOC encoding: the k most frequent sign patterns of alpha.
        codes = np.where(alphas >= 0, 1, -1)
        counts = Counter(map(tuple, codes))
        codebook = np.array([c for c, _ in counts.most_common(k)], dtype=float)
        hamming = (codes[:, None, :] != codebook[None, :, :]).sum(axis=2)
        assignment = np.argmin(hamming, axis=1)

        prototypes = np.empty((codebook.shape[0], k - 1))
        for p in range(codebook.shape[0]):
            members = assignment == p
            prototypes[p] = E[members].mean(axis=0) if members.any() else codebook[p]
        if prototypes.shape[0] < k:
            # Fewer distinct sign patterns than clusters: the kernel bandwidth
            # does not resolve k clusters, and KSC defines no partition.
            raise ValueError(
                f"{name}: only {prototypes.shape[0]} distinct cluster encodings "
                f"were found for n_clusters={k}; the kernel bandwidth (gamma="
                f"{gamma}) does not resolve {k} clusters in this data. Adjust "
                "gamma or reduce n_clusters."
            )

        self._X_fit = X
        self.alphas_ = alphas
        self.bias_ = bias
        self.projections_ = E
        self.codebook_ = codebook
        self.prototypes_ = prototypes
        self.memberships_ = _soft_memberships(E, prototypes)
        return self

    def _predict_memberships(self, X) -> np.ndarray:
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self._X_fit.shape[1], estimator=type(self).__name__)
        E = rbf_kernel(X, self._X_fit, gamma=self.gamma) @ self.alphas_ + self.bias_
        return _soft_memberships(E, self.prototypes_)
