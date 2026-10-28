# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_array, check_generator, check_n_clusters


@typechecked
class SFCMEP(BaseSoftClusterer):
    r"""Semi-supervised fuzzy clustering with a prior membership matrix.

    Builds a prior membership matrix :math:`\tilde U` from a partially
    labelled target vector and an expert-preference parameter :math:`\rho`,
    then alternates

    .. math::

        u_{ij} = \frac{\tilde u_{ij} \exp(-\lVert x_j - v_i\rVert^2/\lambda)}
                      {\sum_l \tilde u_{lj} \exp(-\lVert x_j - v_l\rVert^2/\lambda)},
        \qquad
        v_i = \frac{\sum_j u_{ij} x_j}{\sum_j u_{ij}},

    which is the closed-form minimiser of a k-means data term plus
    :math:`\lambda` times the Kullback--Leibler divergence from the prior
    (with ``lam = 0``, hard c-means). Cluster :math:`i` is initialised at the
    mean of the samples labelled :math:`i`, so cluster indices follow the
    label values.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`C`; labels must lie in ``0 .. C-1``.
    random_state : int, numpy Generator or None, default=None
        Seed for the random fallbacks (unsupervised initialisation and the
        sub-sampling step of the prior construction).
    max_iter : int, default=200
    rho : float, default=0.5
        Expert preference :math:`\rho \in [0, 1]`: the weight given to the
        labelled class in the prior.
    lam : float, default=1.0
        Temperature :math:`\lambda \ge 0` of the exponential distance
        weighting, in units of squared distance.
    tol : float, default=1e-6

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    centers_ : ndarray of shape (n_clusters, n_features)
    prior_memberships_ : ndarray of shape (n_samples, n_clusters)
        The prior :math:`\tilde U` built from the labels.
    n_iter_ : int

    References
    ----------
    S. Xu, Z. Hao, Y. Zhu, Z. Wang, Y. Xiao and B. Liu. Semi-supervised fuzzy
    clustering algorithm based on prior membership degree matrix with expert
    preference. *Expert Systems with Applications*, 238:121812, 2024.
    doi:10.1016/j.eswa.2023.121812

    Notes
    -----
    Earlier releases multiplied by the *previous iteration's* memberships
    instead of the prior, so the prior compounded at every sweep and the
    memberships collapsed to crisp assignments regardless of ``lam``. The
    construction of the prior from the labels follows the reference as
    implemented here but has not been checked against an independent
    implementation.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int | np.random.Generator | None = None,
        max_iter: int = 200,
        rho: float = 0.5,
        lam: float = 1.0,
        tol: float = 1e-6,
    ):
        self.n_clusters = n_clusters
        self.C = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.rho = rho
        self.lam = lam
        self.tol = tol

    def _init_centroids(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers so that cluster i corresponds to class i.
        If y is None (unsupervised), choose random samples instead.
        """
        V_0 = np.zeros((self.C, X.shape[1]))

        if y is None or all(v is None for v in y):  # fully unsupervised
            idx = self.rng.choice(len(X), size=self.C, replace=False)
            V_0 = X[idx]
        else:
            for i in range(self.C):
                # samples of class i
                class_idx = [n for n in range(self.N) if i == y[n]]
                if len(class_idx) > 0:
                    # mean of samples as init center
                    V_0[i] = X[class_idx].mean(axis=0)
                else:
                    # if no sample for this class → random fallback
                    V_0[i] = X[self.rng.integers(len(X))]

        return V_0

    def _init_prior_membership(self, X: np.ndarray, y: np.ndarray, V):
        """
        Simulate the prior membership matrix based on ground-truth labels (y),
        initial cluster centers (V0), and expert preference rho.
        """
        u_org = np.full((self.C, self.N), np.nan)
        labeled_idx = []

        for j in range(self.N):
            if y[j] is None:  # unlabeled
                continue
            labeled_idx.append(j)
            dists = np.linalg.norm(V - X[j], axis=1)
            d_min = dists.min()
            # d_min / d_i, with its limit 1 where the sample sits on centroid i.
            # A class with a single labelled sample has that sample as its
            # initial centroid, so d_i = 0 is the common case, not an edge
            # case; computed directly it gave 0/0 = NaN, which propagated
            # through rho_hat into the whole prior and every membership.
            ratios = np.ones_like(dists)
            positive = dists > 0
            ratios[positive] = d_min / dists[positive]
            for i in range(self.C):
                value = ratios[i]
                if y[j] == i:
                    u_org[i, j] = 1e-6 + self.rho * value
                else:
                    u_org[i, j] = (1 - self.rho) * value
                if u_org[i, j] > 1:
                    u_org[i, j] = 1

        if labeled_idx:
            # Restricted to the labelled columns: the unlabelled ones are
            # entirely NaN, so nanmean already skipped them. Same result,
            # without the all-NaN-slice warnings.
            labelled = u_org[:, labeled_idx]
            M = np.mean(np.max(labelled, axis=0))
            m = np.mean(np.min(labelled, axis=0))
            rho_hat = 0.5 * (M + (1 - (self.C - 1) * m))
        else:
            # No supervision at all: the prior carries no information, so fall
            # back to a uniform membership. Deriving rho_hat from an empty set
            # would make it NaN and poison every subsequent update.
            rho_hat = 1.0 / self.C
        u_org[np.isnan(u_org)] = rho_hat

        if labeled_idx:
            n_subset = max(1, len(labeled_idx) // 8)
            chosen_idx = self.rng.choice(labeled_idx, size=n_subset, replace=False)

            for j in chosen_idx:
                i_max = np.argmax(u_org[:, j])  # strongest cluster
                for i in range(self.C):
                    if i != i_max:
                        u_org[i, j] = rho_hat  # replace other clusters with baseline
        return u_org

    def _update_centroids(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Update cluster centers V given membership matrix U.
        """
        V = np.zeros((self.C, X.shape[1]))
        for i in range(self.C):
            # Membership-weighted mean of the samples. The sum must run over
            # samples only (axis 0 of the stacked (N, D) products); summing
            # over every axis collapses it to a scalar, which then broadcasts
            # into all D coordinates and gives a centroid whose components are
            # all equal.
            V[i] = (U[i] @ X) / (U[i].sum() + 1e-8)
        return V

    def _update_membership_matrix(
        self, X: np.ndarray, U: np.ndarray, V: np.ndarray
    ) -> np.ndarray:
        """Re-weight the cluster-major matrix ``U`` by ``exp(-d^2 / lam)``.

        With ``lam = 0`` this is the hard c-means assignment. The exponents
        are shifted by their per-sample maximum before exponentiating, which
        cancels in the normalisation and keeps a sample far from every
        centroid from underflowing to an all-zero column. A zero prior entry
        is floored at the smallest positive double, so a column can never be
        all zero.

        Earlier releases replaced a zero prior entry by one, the largest
        prior weight. A zero arises for the non-labelled classes of a
        labelled sample when ``rho = 1``, so full trust in the labels gave
        those classes the most weight and enforced the labels less than
        ``rho = 0.9`` did.
        """
        d2 = np.sum((X[None, :, :] - V[:, None, :]) ** 2, axis=2)  # (C, N)
        if self.lam == 0:
            U_new = np.zeros((self.C, self.N))
            U_new[np.argmin(d2, axis=0), np.arange(self.N)] = 1.0
            return U_new
        logits = -d2 / self.lam
        logits -= logits.max(axis=0, keepdims=True)
        prior = np.maximum(U, np.finfo(np.float64).tiny)
        num = prior * np.exp(logits)
        return num / (num.sum(axis=0, keepdims=True) + 1e-12)

    @staticmethod
    def _normalise_labels(y, n_samples: int, n_clusters: int, name: str):
        """Object array of int labels with ``None`` for unlabelled samples.

        ``None``, ``NaN`` and negative values (scikit-learn's ``-1``
        convention) all mean "unlabelled".
        """
        y = np.asarray(y, dtype=object).ravel()
        if y.shape[0] != n_samples:
            raise ValueError(
                f"{name}: y has {y.shape[0]} entries but X has {n_samples} samples."
            )
        out = np.full(n_samples, None, dtype=object)
        for i, value in enumerate(y):
            if value is None:
                continue
            try:
                f = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: label {value!r} is not numeric.") from exc
            if np.isnan(f) or f < 0:
                continue
            if f != int(f) or int(f) >= n_clusters:
                raise ValueError(
                    f"{name}: labels must be integers in [0, {n_clusters - 1}] "
                    f"(or -1/None/NaN for unlabelled); got {value!r}."
                )
            out[i] = int(f)
        return out

    def fit_predict(self, X: np.ndarray, y) -> np.ndarray:
        """Cluster ``X`` guided by the partial labels ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class index in ``0 .. n_clusters-1`` for labelled samples and
            ``None``, ``NaN`` or ``-1`` for unlabelled ones.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
            The membership matrix.
        """
        name = type(self).__name__
        X = check_array(X, estimator=name)
        self.C = check_n_clusters(self.n_clusters, X.shape[0], estimator=name)
        if self.lam < 0:
            raise ValueError(f"{name}: lam must be >= 0; got {self.lam}.")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"{name}: rho must lie in [0, 1]; got {self.rho}.")
        y = self._normalise_labels(y, X.shape[0], self.C, name)
        self.rng = check_generator(self.random_state)
        self.N = X.shape[0]
        V_0 = self._init_centroids(X, y)
        U_prior = self._init_prior_membership(X, y, V_0)
        U_prev = self._update_membership_matrix(X, U_prior, V_0)
        V_prev = V_0.copy()

        # Every sweep re-weights the *prior*, not the previous memberships:
        # u_ij is proportional to prior_ij * exp(-d_ij^2 / lam).
        for t in range(self.max_iter):
            V = self._update_centroids(X, U_prev)
            U = self._update_membership_matrix(X, U_prior, V)
            # Convergence check
            if (
                np.linalg.norm(U - U_prev) <= self.tol
                and np.linalg.norm(V - V_prev) <= self.tol
            ):
                break
            U_prev = U
            V_prev = V

        # Publish under the names the estimator protocol searches, so that
        # memberships_, labels_, centers_ and n_clusters are populated like
        # every other estimator. See soft_clustering/_base.py.
        self.centroids = V
        self.centers_ = V
        self.membership_matrix = U.T
        self.memberships_ = U.T
        self.prior_memberships_ = U_prior.T
        self.n_iter_ = t + 1
        return self.memberships_
