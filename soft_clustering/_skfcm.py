# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Spatially constrained kernelized fuzzy c-means (Zhang and Chen, 2004)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve
from typeguard import typechecked

from ._base import BaseSoftClusterer, ratio_memberships
from ._fcm import _euclidean_dist2, _init_centers_kpp
from ._validation import check_array, check_generator, check_n_clusters, check_scalar

_NEIGHBOURS = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])


@typechecked
class SKFCM(BaseSoftClusterer):
    r"""Spatially constrained kernelized fuzzy c-means (SKFCM) for images.

    Kernelized fuzzy c-means with a Gaussian kernel
    :math:`K(x, v) = \exp(-\gamma\lVert x - v\rVert^2)` plus a penalty that
    favours memberships agreeing with those of the 8-connected neighbours
    :math:`N_j` of every pixel (Zhang and Chen, 2004):

    .. math::

        J = \sum_{j,k} u_{jk}^m \big(1 - K(x_j, v_k)\big)
          + \frac{\alpha}{N_R} \sum_{j,k} u_{jk}^m \sum_{r \in N_j} (1 - u_{rk})^m .

    The updates are

    .. math::

        u_{jk} \propto \Big(1 - K(x_j, v_k)
            + \frac{\alpha}{N_R} \sum_{r \in N_j} (1 - u_{rk})^m\Big)^{-1/(m-1)},
        \qquad
        v_k = \frac{\sum_j u_{jk}^m K(x_j, v_k)\, x_j}{\sum_j u_{jk}^m K(x_j, v_k)} ,

    where :math:`N_R` is the number of neighbours (fewer at the image
    border). The neighbourhood term is evaluated with the memberships of the
    previous sweep, as in the reference.

    Parameters
    ----------
    n_clusters : int, default=3
    m : float, default=2.0
        Fuzzifier, ``m > 1``.
    gamma : float, default=1.0
        Kernel parameter :math:`\gamma > 0`, in units of ``1/||X||^2``.
    lambda_ : float, default=0.5
        Spatial weight :math:`\alpha \ge 0`.
    max_iter : int, default=100
    tol : float, default=1e-5
        Stop when no membership changes by more than ``tol``.
    random_state : int, numpy Generator or None, default=None
        Seed for the k-means++ initialisation of the prototypes.

    Attributes
    ----------
    memberships_ : ndarray of shape (H * W, n_clusters)
    centers_ : ndarray of shape (n_clusters, n_features)
    n_iter_ : int

    References
    ----------
    D.-Q. Zhang and S.-C. Chen. A novel kernelized fuzzy c-means algorithm
    with application in medical image segmentation. *Artificial Intelligence
    in Medicine*, 32(1):37--50, 2004. doi:10.1016/j.artmed.2004.01.012

    Notes
    -----
    Earlier releases measured distances to implicit kernel-space centroids,
    which requires the dense ``(H*W) x (H*W)`` kernel matrix (34 GB for a
    256 x 256 image), and used ``1 - mean neighbour membership`` as the
    spatial term. The present implementation is the reference formulation
    and needs :math:`O(H W K)` memory.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        gamma: float = 1.0,
        lambda_: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, shape: tuple[int, int]) -> SKFCM:
        """Fit on pixel features.

        Parameters
        ----------
        X : array-like of shape (H * W, n_features)
            Pixel features in row-major (C) order, e.g.
            ``image.reshape(-1, 1)`` for a grayscale image.
        shape : tuple of int
            The image shape ``(H, W)``.
        """
        name = type(self).__name__
        X = check_array(X, estimator=name)
        H, W = shape
        if H * W != X.shape[0]:
            raise ValueError(
                f"{name}: shape {shape} has {H * W} pixels but X has "
                f"{X.shape[0]} rows."
            )
        K = check_n_clusters(self.n_clusters, X.shape[0], estimator=name)
        m = check_scalar(self.m, "m", estimator=name, lower=1.0, lower_inclusive=False)
        gamma = check_scalar(
            self.gamma, "gamma", estimator=name, lower=0.0, lower_inclusive=False
        )
        alpha = check_scalar(self.lambda_, "lambda_", estimator=name, lower=0.0)

        count = convolve(np.ones((H, W)), _NEIGHBOURS, mode="constant", cval=0.0)
        count = np.maximum(count, 1.0).ravel()

        V = _init_centers_kpp(X, K, check_generator(self.random_state))
        Kmat = np.exp(-gamma * _euclidean_dist2(X, V))
        U = ratio_memberships(np.maximum(1.0 - Kmat, 1e-300), 1.0 / (m - 1.0))
        for it in range(self.max_iter):
            U_prev = U
            spatial = np.empty_like(U)
            for k in range(K):
                field = ((1.0 - U[:, k]) ** m).reshape(H, W)
                spatial[:, k] = convolve(
                    field, _NEIGHBOURS, mode="constant", cval=0.0
                ).ravel()
            spatial /= count[:, None]

            weights = U**m * Kmat
            denom = weights.sum(axis=0)
            alive = denom > 1e-300
            V[alive] = (weights[:, alive].T @ X) / denom[alive, None]
            Kmat = np.exp(-gamma * _euclidean_dist2(X, V))
            D = np.maximum(1.0 - Kmat + alpha * spatial, 1e-300)
            U = ratio_memberships(D, 1.0 / (m - 1.0))
            if np.max(np.abs(U - U_prev)) < self.tol:
                break

        self._n_samples_hint = X.shape[0]
        self.image_shape_ = (H, W)
        self.centers_ = V
        self.memberships_ = U
        self.U = U
        self.n_iter_ = it + 1
        return self
