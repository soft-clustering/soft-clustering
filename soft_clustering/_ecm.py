# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Evidential c-means (Masson and Denoeux, 2008)."""

from __future__ import annotations

import itertools

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._fcm import _euclidean_dist2, _init_centers_kpp
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_n_features,
    check_scalar,
    objective_converged,
)

#: Above this many clusters the full power set (2**K - 1 focal sets) is
#: refused unless requested explicitly; memory and time grow as 2**K.
_MAX_FULL_CLUSTERS = 12


def _focal_sets(n_clusters: int, kind: str) -> np.ndarray:
    """Boolean ``(n_focal, n_clusters)`` matrix of the non-empty focal sets.

    ``"full"``: every non-empty subset of the frame, ordered by cardinality
    and then lexicographically. ``"simple"``: the singletons and the whole
    frame :math:`\\Omega`.
    """
    if kind == "full":
        rows = []
        for size in range(1, n_clusters + 1):
            for combo in itertools.combinations(range(n_clusters), size):
                row = np.zeros(n_clusters, dtype=bool)
                row[list(combo)] = True
                rows.append(row)
        return np.array(rows)
    if kind == "simple":
        F = np.eye(n_clusters, dtype=bool)
        if n_clusters > 1:
            F = np.vstack([F, np.ones((1, n_clusters), dtype=bool)])
        return F
    raise ValueError(f"focal_sets must be 'full' or 'simple'; got {kind!r}.")


def _ecm_masses(D, card, alpha, beta, delta2):
    """Mass update (Masson and Denoeux, 2008, Eq. 13), evaluated stably.

    ``m_ij`` is proportional to ``(|A_j|^alpha d_ij^2)^{-1/(beta-1)}`` and the
    empty set receives ``delta^{-2/(beta-1)}``, all normalised to one. The
    per-row maximum is factored out in log space, so neither a sample on a
    prototype (zero distance) nor a large exponent overflows.
    """
    p = 1.0 / (beta - 1.0)
    tiny = np.finfo(np.float64).tiny
    log_w = -p * (alpha * np.log(card)[None, :] + np.log(np.maximum(D, tiny)))
    log_empty = np.full((D.shape[0], 1), -p * np.log(delta2))
    log_all = np.hstack([log_empty, log_w])
    log_all -= log_all.max(axis=1, keepdims=True)
    w = np.exp(log_all)
    w /= w.sum(axis=1, keepdims=True)
    return w[:, 1:], w[:, 0]


@typechecked
class ECM(BaseSoftClusterer):
    r"""Evidential c-means (ECM).

    ECM produces a *credal partition*: every sample receives a mass function
    over the subsets of the set of clusters :math:`\Omega = \{\omega_1, \dots,
    \omega_K\}`. Mass on a singleton expresses membership in one cluster, mass
    on a larger set :math:`A_j` expresses ambiguity between the clusters it
    contains, and mass on the empty set flags the sample as an outlier.

    Each non-empty focal set :math:`A_j` is represented by the barycentre of
    the prototypes it contains, :math:`\bar v_j = |A_j|^{-1}\sum_{k \in A_j}
    v_k`, and ECM minimises

    .. math::

        J = \sum_i \sum_{A_j \ne \emptyset} |A_j|^{\alpha}\, m_{ij}^{\beta}\,
            \lVert x_i - \bar v_j\rVert^2 + \sum_i \delta^2\, m_{i\emptyset}^{\beta}

    subject to :math:`\sum_j m_{ij} + m_{i\emptyset} = 1`. The masses have a
    closed-form update, and the prototypes solve the :math:`K \times K` linear
    system :math:`H V = B` of Masson and Denoeux (2008, Eqs. 13--20). Both
    steps decrease :math:`J`, so ``objective_trajectory_`` is
    non-increasing.

    ``memberships_`` is the pignistic transform of the credal partition,
    :math:`\mathrm{BetP}_i(\omega_k) = \sum_{A_j \ni \omega_k}
    m_{ij} / \big(|A_j| (1 - m_{i\emptyset})\big)`, which sums to one.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`K`.
    m : float, default=2.0
        Exponent :math:`\beta > 1` applied to the masses (the analogue of the
        fuzzifier; Masson and Denoeux recommend 2).
    alpha : float, default=1.0
        Exponent :math:`\alpha \ge 0` of the cardinality penalty. Larger
        values discourage mass on imprecise (multi-cluster) focal sets.
    delta : float, default=10.0
        Distance :math:`\delta > 0` of every sample to the empty set: samples
        farther than about ``delta`` from all prototypes are treated as
        outliers. It is in the units of ``X``.
    focal_sets : {"full", "simple"}, default="full"
        ``"full"`` uses all :math:`2^K - 1` non-empty subsets, as in the
        reference. ``"simple"`` restricts them to the singletons and
        :math:`\Omega`, which Masson and Denoeux recommend for large
        :math:`K`. ``"full"`` is refused above 12 clusters.
    max_iter : int, default=100
        Maximum number of sweeps.
    tol : float, default=1e-5
        Stop when :math:`J` changes by at most ``tol`` relative to its
        magnitude.
    init : {"kmeans++"} or ndarray of shape (n_clusters, n_features)
        Prototype initialisation: k-means++ seeding, or explicit initial
        prototypes.
    random_state : int, numpy Generator or None, default=None
        Seed for the k-means++ initialisation.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Pignistic probabilities (rows sum to one).
    masses_ : ndarray of shape (n_samples, 1 + n_focal)
        The credal partition. Column 0 is the mass of the empty set; column
        ``j + 1`` is the mass of ``focal_sets_[j]``. Rows sum to one.
    focal_sets_ : ndarray of bool, shape (n_focal, n_clusters)
        Membership of each cluster in each non-empty focal set.
    centers_ : ndarray of shape (n_clusters, n_features)
        Cluster prototypes (also ``prototypes``).
    objective_trajectory_ : ndarray
    n_iter_ : int

    References
    ----------
    M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy
    c-means algorithm. *Pattern Recognition*, 41(4):1384--1397, 2008.
    doi:10.1016/j.patcog.2007.08.014

    Notes
    -----
    Earlier releases implemented only the singleton focal sets plus the empty
    set, which is noise clustering rather than ECM. The full credal partition
    is validated against ``evclust``, the reference package of the method's
    authors, in ``tests/test_external_agreement.py``.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _supports_out_of_sample = True

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        alpha: float = 1.0,
        delta: float = 10.0,
        focal_sets: str = "full",
        max_iter: int = 100,
        tol: float = 1e-5,
        init: str | np.ndarray = "kmeans++",
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.delta = delta
        self.focal_sets = focal_sets
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    # ------------------------------------------------------------------

    def _check_params(self, n_samples: int) -> tuple[int, float, float, float]:
        name = type(self).__name__
        K = check_n_clusters(self.n_clusters, n_samples, estimator=name)
        beta = check_scalar(
            self.m, "m", estimator=name, lower=1.0, lower_inclusive=False
        )
        alpha = check_scalar(self.alpha, "alpha", estimator=name, lower=0.0)
        delta = check_scalar(
            self.delta, "delta", estimator=name, lower=0.0, lower_inclusive=False
        )
        if self.focal_sets == "full" and K > _MAX_FULL_CLUSTERS:
            raise ValueError(
                f"{name}: focal_sets='full' needs 2**{K} - 1 focal sets; use "
                f"focal_sets='simple' for more than {_MAX_FULL_CLUSTERS} clusters."
            )
        return K, beta, alpha, delta

    def _distances(self, X, V):
        F = self.focal_sets_
        card = F.sum(axis=1).astype(np.float64)
        barycentres = (F.astype(np.float64) @ V) / card[:, None]
        return _euclidean_dist2(X, barycentres), card

    def fit(self, X) -> ECM:
        """Fit the model on a feature matrix of shape ``(n_samples, n_features)``."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        n, d = X.shape
        K, beta, alpha, delta = self._check_params(n)
        delta2 = delta**2
        self.focal_sets_ = _focal_sets(K, self.focal_sets)
        F = self.focal_sets_.astype(np.float64)

        if isinstance(self.init, np.ndarray):
            V = check_array(self.init, estimator=name, name="init")
            if V.shape != (K, d):
                raise ValueError(
                    f"{name}: init has shape {V.shape}; expected {(K, d)}."
                )
        elif self.init == "kmeans++":
            V = _init_centers_kpp(X, K, check_generator(self.random_state))
        else:
            raise ValueError(
                f"{name}: init must be 'kmeans++' or an array of initial "
                f"prototypes; got {self.init!r}."
            )

        # Focal sets containing cluster l, and containing both k and l.
        contains = F.T  # (K, n_focal)

        trajectory = []
        J_prev = np.inf
        for _ in range(self.max_iter):
            D, card = self._distances(X, V)
            M, M_empty = _ecm_masses(D, card, alpha, beta, delta2)
            Mb = M**beta

            # Prototype update, H V = B (Masson and Denoeux, 2008, Eqs. 18-20).
            weights_B = Mb * card[None, :] ** (alpha - 1.0)  # (n, n_focal)
            B = (contains @ weights_B.T) @ X  # (K, d)
            mass_per_focal = Mb.sum(axis=0) * card ** (alpha - 2.0)
            H = (contains * mass_per_focal[None, :]) @ contains.T  # (K, K)
            try:
                V = np.linalg.solve(H, B)
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(H, B, rcond=None)[0]

            D, card = self._distances(X, V)
            M, M_empty = _ecm_masses(D, card, alpha, beta, delta2)
            J = float(
                np.sum((card[None, :] ** alpha) * (M**beta) * D)
                + delta2 * np.sum(M_empty**beta)
            )
            trajectory.append(J)
            if objective_converged(J_prev, J, self.tol):
                break
            J_prev = J

        self.centers_ = V
        self.masses_ = np.hstack([M_empty[:, None], M])
        self.memberships_ = self._pignistic(M, M_empty)
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        self.n_iter_ = len(trajectory)
        return self

    def _pignistic(self, M, M_empty):
        F = self.focal_sets_.astype(np.float64)
        card = F.sum(axis=1)
        betp = (M / card[None, :]) @ F
        total = betp.sum(axis=1, keepdims=True)
        # Equivalent to dividing by 1 - m(empty), but robust when a sample
        # far from every prototype puts almost all its mass on the empty set.
        K = F.shape[1]
        return np.where(total > 0, betp / np.where(total > 0, total, 1.0), 1.0 / K)

    def _predict_memberships(self, X):
        X = check_array(X, estimator=type(self).__name__)
        check_n_features(X, self.centers_.shape[1], estimator=type(self).__name__)
        D, card = self._distances(X, self.centers_)
        M, M_empty = _ecm_masses(D, card, self.alpha, self.m, self.delta**2)
        return self._pignistic(M, M_empty)

    @property
    def prototypes(self) -> np.ndarray | None:
        """Alias of ``centers_``, kept for backward compatibility."""
        return getattr(self, "centers_", None)

    @property
    def mass(self) -> np.ndarray | None:
        """Alias of ``masses_``, kept for backward compatibility."""
        return getattr(self, "masses_", None)

    def get_membership(self) -> np.ndarray:
        """Return the credal partition ``masses_``.

        Column 0 holds the mass of the empty set and column ``j + 1`` the mass
        of ``focal_sets_[j]``.
        """
        self._check_fitted()
        return self.masses_
