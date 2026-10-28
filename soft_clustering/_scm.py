# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Subtractive clustering (Chiu, 1994)."""

from __future__ import annotations

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_array, check_n_features, check_scalar


@typechecked
class SCM(BaseSoftClusterer):
    r"""Subtractive clustering (Chiu, 1994).

    Every sample is a candidate cluster center. After scaling each feature to
    ``[0, 1]``, the potential of sample :math:`x_i` is

    .. math::

        P_i = \sum_j \exp(-\alpha \lVert x_i - x_j\rVert^2),
        \qquad \alpha = 4 / r_a^2 .

    The sample with the highest potential :math:`P^*` becomes a center and
    the potentials are reduced by :math:`P^* \exp(-\beta\lVert x_i -
    x^*\rVert^2)` with :math:`\beta = 4/r_b^2`, :math:`r_b = 1.5\, r_a`. The
    process repeats with Chiu's acceptance rule, where :math:`P_1` is the
    potential of the first center:

    * :math:`P^* > \bar\epsilon P_1`: accept :math:`x^*` as a center;
    * :math:`P^* < \underline\epsilon P_1`: stop;
    * otherwise (the gray zone) accept :math:`x^*` only if
      :math:`d_{\min}/r_a + P^*/P_1 \ge 1`, where :math:`d_{\min}` is its
      distance to the nearest accepted center; if rejected, set its potential
      to zero and test the next-highest candidate.

    The number of clusters is therefore discovered, not given. Memberships are
    the Gaussian neighbourhood degrees :math:`\exp(-\alpha\lVert x_i -
    c_k\rVert^2)` normalised across centers, so rows sum to one.

    Time is :math:`O(n^2)` (the pairwise potential); memory is :math:`O(n)`,
    because the potentials are accumulated in blocks of rows.

    Parameters
    ----------
    ra : float, default=0.5
        Neighbourhood radius :math:`r_a > 0`, in the unit-scaled feature space.
    ea : float, default=0.5
        Accept ratio :math:`\bar\epsilon`.
    er : float, default=0.15
        Reject ratio :math:`\underline\epsilon`, ``0 <= er <= ea``.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, K)
    centers_ : ndarray of shape (K, n_features)
        Centers in the original feature units.
    center_indices_ : ndarray of int
        Indices of the samples chosen as centers.

    References
    ----------
    S. L. Chiu. Fuzzy model identification based on cluster estimation.
    *Journal of Intelligent and Fuzzy Systems*, 2(3):267--278, 1994.
    doi:10.3233/IFS-1994-2306

    Notes
    -----
    Earlier releases implemented only the accept test and ignored ``er``.
    """

    _determines_k = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(self, ra: float = 0.5, ea: float = 0.5, er: float = 0.15):
        self.ra = ra
        self.ea = ea
        self.er = er

    def fit(self, X) -> np.ndarray:
        """Fit on a feature matrix and return the centers."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        ra = check_scalar(
            self.ra, "ra", estimator=name, lower=0.0, lower_inclusive=False
        )
        ea = check_scalar(self.ea, "ea", estimator=name, lower=0.0, upper=1.0)
        er = check_scalar(self.er, "er", estimator=name, lower=0.0, upper=ea)

        X_min = X.min(axis=0)
        span = X.max(axis=0) - X_min
        span[span == 0] = 1.0
        Xn = (X - X_min) / span
        sq = np.sum(Xn**2, axis=1)

        def dist_sq_to(rows):
            """Squared distances from the samples ``rows`` to every sample."""
            return np.maximum(sq[rows, None] + sq[None, :] - 2.0 * Xn[rows] @ Xn.T, 0.0)

        alpha = 4.0 / ra**2
        beta = 4.0 / (1.5 * ra) ** 2
        # The potentials need every pairwise distance, but not all at once:
        # blocks of rows keep the memory O(block * n) instead of O(n^2).
        n = Xn.shape[0]
        block = max(1, min(n, 2_000_000 // max(n, 1)))
        potentials = np.empty(n)
        for start in range(0, n, block):
            rows = np.arange(start, min(start + block, n))
            potentials[rows] = np.exp(-alpha * dist_sq_to(rows)).sum(axis=1)
        p1 = float(potentials.max())

        chosen: list[int] = []
        while True:
            idx = int(np.argmax(potentials))
            p = float(potentials[idx])
            if p <= 0.0:
                break
            if p > ea * p1:
                accept = True
            elif p < er * p1:
                break
            else:
                d_min = (
                    float(np.sqrt(dist_sq_to(np.array([idx]))[0, chosen].min()))
                    if chosen
                    else np.inf
                )
                accept = d_min / ra + p / p1 >= 1.0
            if not accept:
                potentials[idx] = 0.0
                continue
            chosen.append(idx)
            potentials = potentials - p * np.exp(-beta * dist_sq_to(np.array([idx]))[0])

        self.center_indices_ = np.array(chosen, dtype=int)
        self.centers_ = X[self.center_indices_].copy()
        self._X_min, self._span, self._alpha = X_min, span, alpha
        self.memberships_ = self._predict_from(Xn)
        return self.centers_

    def _predict_from(self, Xn):
        Cn = (self.centers_ - self._X_min) / self._span
        d2 = np.maximum(
            np.sum(Xn**2, axis=1, keepdims=True)
            + np.sum(Cn**2, axis=1)
            - 2.0 * Xn @ Cn.T,
            0.0,
        )
        # Shift by the per-row minimum before exponentiating; it cancels in
        # the normalisation and prevents all-zero rows far from every center.
        logits = -self._alpha * (d2 - d2.min(axis=1, keepdims=True))
        U = np.exp(logits)
        return U / U.sum(axis=1, keepdims=True)

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        return self._predict_from((X - self._X_min) / self._span)
