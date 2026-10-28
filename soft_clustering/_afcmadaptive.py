# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Adaptive fuzzy c-means for images with intensity inhomogeneity (Pham and Prince, 1999)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typeguard import typechecked

from ._base import BaseSoftClusterer, ratio_memberships
from ._validation import check_array, check_n_clusters, check_scalar


def _difference_operators(H: int, W: int):
    """Forward-difference operators along rows and columns, zero at the far edge."""

    def d1(n):
        if n == 1:
            return sp.csr_matrix((1, 1))
        main = -np.ones(n)
        main[-1] = 0.0
        return sp.diags([main, np.ones(n - 1)], [0, 1], format="csr")

    Dx = sp.kron(sp.identity(H), d1(W), format="csr")  # along a row
    Dy = sp.kron(d1(H), sp.identity(W), format="csr")  # down a column
    return Dx, Dy


@typechecked
class AFCMAdaptive(BaseSoftClusterer):
    r"""Adaptive fuzzy c-means for image segmentation (Pham and Prince, 1999).

    Segments a grayscale image whose intensity is corrupted by a smooth,
    multiplicative gain field (intensity inhomogeneity, e.g. MR bias). The
    observed intensity at pixel :math:`j` is modelled as :math:`y_j \approx
    g_j v_k` for the class :math:`k` it belongs to, and the objective is

    .. math::

        J = \sum_{j,k} u_{jk}^q (y_j - g_j v_k)^2
          + \lambda_1 \sum_r \lVert D_r g\rVert^2
          + \lambda_2 \sum_{r,s} \lVert D_r D_s g\rVert^2 ,

    where :math:`D_r` are finite-difference operators along the image axes,
    so the first penalty keeps the gain field slowly varying and the second
    keeps it smooth. Block coordinate descent gives

    .. math::

        u_{jk} \propto (y_j - g_j v_k)^{-2/(q-1)},\qquad
        v_k = \frac{\sum_j u_{jk}^q g_j y_j}{\sum_j u_{jk}^q g_j^2},

    and the gain field solves the sparse symmetric positive-definite system

    .. math::

        \Big(\mathrm{diag}\big(\textstyle\sum_k u_{jk}^q v_k^2\big)
             + \lambda_1 H_1 + \lambda_2 H_2\Big)\, g
        = y \odot \textstyle\sum_k u_{jk}^q v_k ,

    with :math:`H_1 = \sum_r D_r^\top D_r` and :math:`H_2 = \sum_{r,s}
    (D_r D_s)^\top D_r D_s` (Pham and Prince, 1999, Eqs. 6--10). Each block
    update minimises :math:`J`, so ``objective_trajectory_`` is
    non-increasing. Pham and Prince solve the gain system with multigrid;
    SCPP uses a sparse direct solver, which is exact but scales less well
    to very large images.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of tissue classes.
    m : float, default=2.0
        Fuzzifier :math:`q > 1`.
    k1 : float, default=1.0
        First-order smoothness weight :math:`\lambda_1 \ge 0`.
    k2 : float, default=1.0
        Second-order smoothness weight :math:`\lambda_2 \ge 0`. Both weights
        are relative to the squared intensity scale of the image.
    max_iter : int, default=100
    tol : float, default=1e-4
        Stop when no membership changes by more than ``tol``.

    Attributes
    ----------
    memberships_ : ndarray of shape (H * W, n_clusters)
        Pixel-major memberships (rows sum to one).
    centers_ : ndarray of shape (n_clusters, 1)
        Class intensities :math:`v_k`.
    gain_field_ : ndarray of shape (H, W)
        Estimated multiplicative gain :math:`g` (also ``multiplier``).
    image_shape_ : tuple
    objective_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    D. L. Pham and J. L. Prince. An adaptive fuzzy C-means algorithm for
    image segmentation in the presence of intensity inhomogeneities.
    *Pattern Recognition Letters*, 20(1):57--68, 1999.
    doi:10.1016/S0167-8655(98)00121-4

    Notes
    -----
    Earlier releases computed the class intensities as
    :math:`\sum u y / \sum u y^2` (approximately the reciprocal of the
    intensity) and updated the gain with a Gaussian blur in place of the
    derivative penalties; they did not segment even a noise-free two-level
    image. The present implementation is the reference formulation.

    The model has a scale ambiguity (:math:`g \to c g`, :math:`v \to v/c`
    leaves the data term unchanged). The gain is initialised to one and is
    not renormalised, so that every step stays a descent step.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)

    # Fitted attributes, set by ``fit`` (not before).
    centers: np.ndarray
    membership: np.ndarray
    multiplier: np.ndarray

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        k1: float = 1.0,
        k2: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.k1 = k1
        self.k2 = k2
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, image) -> AFCMAdaptive:
        """Fit on a 2-D grayscale image of shape ``(H, W)``."""
        name = type(self).__name__
        image = check_array(image, estimator=name, name="image")
        H, W = image.shape
        n = H * W
        K = check_n_clusters(self.n_clusters, n, estimator=name)
        q = check_scalar(self.m, "m", estimator=name, lower=1.0, lower_inclusive=False)
        lam1 = check_scalar(self.k1, "k1", estimator=name, lower=0.0)
        lam2 = check_scalar(self.k2, "k2", estimator=name, lower=0.0)

        y = image.ravel()
        Dx, Dy = _difference_operators(H, W)
        H1 = (Dx.T @ Dx + Dy.T @ Dy).tocsr()
        second = [Dx @ Dx, Dx @ Dy, Dy @ Dx, Dy @ Dy]
        H2 = sum((D.T @ D for D in second), sp.csr_matrix((n, n))).tocsr()
        penalty = (lam1 * H1 + lam2 * H2).tocsr()

        # Initialise the class intensities at evenly spaced quantiles of the
        # image and the gain at one.
        v = np.quantile(y, (np.arange(K) + 0.5) / K).astype(np.float64)
        if np.unique(v).size < K:
            v = np.linspace(y.min(), y.max(), K) + np.arange(K) * 1e-9
        g = np.ones(n)
        U = ratio_memberships(
            (y[:, None] - g[:, None] * v[None, :]) ** 2, 1.0 / (q - 1.0)
        )

        trajectory = []
        for it in range(self.max_iter):
            U_prev = U
            Uq = U**q
            # Class intensities.
            v = (Uq * (g * y)[:, None]).sum(axis=0) / np.maximum(
                (Uq * (g**2)[:, None]).sum(axis=0), 1e-300
            )
            # Gain field.
            diag = (Uq * v[None, :] ** 2).sum(axis=1)
            rhs = y * (Uq * v[None, :]).sum(axis=1)
            A = (sp.diags(np.maximum(diag, 1e-12)) + penalty).tocsc()
            g = np.asarray(spsolve(A, rhs)).ravel()
            # Memberships.
            resid = (y[:, None] - g[:, None] * v[None, :]) ** 2
            U = ratio_memberships(resid + 1e-300, 1.0 / (q - 1.0))

            J = float(np.sum(U**q * resid) + g @ (penalty @ g))
            trajectory.append(J)
            if np.max(np.abs(U - U_prev)) < self.tol:
                break

        self.image_shape_ = (H, W)
        self._n_samples_hint = n
        self.memberships_ = U
        self.membership = U.reshape(H, W, K)
        self.centers = v
        self.centers_ = v.reshape(-1, 1)
        self.multiplier = g.reshape(H, W)
        self.gain_field_ = self.multiplier
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = it + 1
        return self

    def label_map(self) -> np.ndarray:
        """Hard segmentation in image layout, shape ``(H, W)``.

        ``predict()`` returns the flat ``(H*W,)`` labelling required by the
        estimator protocol; this is the same assignment reshaped back to the
        image grid.
        """
        self._check_fitted()
        return self.labels_.reshape(self.image_shape_)

    def get_membership(self) -> np.ndarray:
        """Membership array in image layout, shape ``(H, W, K)``."""
        self._check_fitted()
        return self.membership
